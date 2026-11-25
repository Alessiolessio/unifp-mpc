#pragma once

#include "world.h"

#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>

namespace unifp {

struct RolloutStatus {
    std::size_t steps = 0;
    std::string failure;
    StepTimes time;
};

class Rollouts {
public:
    Rollouts(const World& live, int workers) : model(live.plant.model) {
        if (workers < 1) throw std::invalid_argument("workers must be positive");
        for (int i = 0; i < workers; ++i) {
            worlds.push_back(std::make_unique<World>(model, live.policy));
        }
        threads.reserve(workers - 1);
        try {
            for (int i = 1; i < workers; ++i) threads.emplace_back(&Rollouts::serve, this, i);
        } catch (...) {
            shutdown();
            throw;
        }
    }

    ~Rollouts() { shutdown(); }

    // Construct the execution counterpart from the compiled nominal model. Only
    // these physical coefficients can differ: every topology index, frame,
    // actuator table, policy and state layout is identical by construction.
    std::unique_ptr<World> execution_world(double mass_scale, double floor_scale, bool known_load,
                                           double ground_scale, double torso_scale) {
        std::lock_guard lock(batch_mutex);
        if (!(std::isfinite(mass_scale) && mass_scale > 0 &&
              std::isfinite(floor_scale) && floor_scale > 0 &&
              std::isfinite(ground_scale) && ground_scale > 0 &&
              std::isfinite(torso_scale) && torso_scale > 0))
            throw std::invalid_argument("expected finite positive physics multipliers");
        auto copy = std::shared_ptr<mjModel>(mj_copyModel(nullptr, model.get()), mj_deleteModel);
        if (!copy) throw std::bad_alloc();
        if (mass_scale != 1.) {
            int body = mj_name2id(copy.get(), mjOBJ_BODY, "box_body");
            if (body < 0) body = mj_name2id(copy.get(), mjOBJ_BODY, "traffic_cone");
            if (body < 1 || copy->body_jntnum[body] != 1 ||
                copy->jnt_type[copy->body_jntadr[body]] != mjJNT_FREE)
                throw std::invalid_argument("expected free box or cone");
            copy->body_mass[body] *= mass_scale;
            for (int k = 0; k < 3; ++k) copy->body_inertia[3*body+k] *= mass_scale;
        }
        if (torso_scale != 1.) {
            const int body = mj_name2id(copy.get(), mjOBJ_BODY, "body");
            if (body < 1) throw std::invalid_argument("expected Spot torso");
            copy->body_mass[body] *= torso_scale;
            for (int k = 0; k < 3; ++k) copy->body_inertia[3*body+k] *= torso_scale;
        }
        if (mass_scale != 1. || torso_scale != 1.) {
            DataPtr data(mj_makeData(copy.get()), mj_deleteData);
            if (!data) throw std::bad_alloc();
            mj_setConst(copy.get(), data.get());
        }
        if (floor_scale != 1.) {
            const int pair = mj_name2id(copy.get(), mjOBJ_PAIR, "box_ground");
            if (pair < 0) throw std::invalid_argument("expected explicit box_ground contact pair");
            for (int k = 0; k < 2; ++k) copy->pair_friction[5*pair+k] *= floor_scale;
        }
        if (ground_scale != 1.) {
            const int geom = mj_name2id(copy.get(), mjOBJ_GEOM, "ground");
            if (geom < 0) throw std::invalid_argument("expected ground geom");
            copy->geom_friction[3*geom] *= ground_scale;
        }
        auto live = std::make_unique<World>(copy, worlds.front()->policy);
        execution_model = copy;
        different_physics = mass_scale != 1. || floor_scale != 1. ||
                            ground_scale != 1. || torso_scale != 1.;
        copy_applied_load = known_load;
        return live;
    }

    bool mismatched() {
        std::lock_guard lock(batch_mutex);
        return different_physics;
    }

    // Between episodes keep the inference session and idle threads. A new immutable
    // model identity also prevents an old snapshot from restoring stale physics.
    std::unique_ptr<World> new_episode(const std::string& scene) {
        std::lock_guard batch_lock(batch_mutex);
        auto next = load_model(scene);
        auto policy = worlds.front()->policy;
        auto live = std::make_unique<World>(next, policy);
        std::vector<std::unique_ptr<World>> fresh;
        for (std::size_t i = 0; i < worlds.size(); ++i) {
            fresh.push_back(std::make_unique<World>(next, policy));
        }
        worlds.swap(fresh);
        model = next;
        execution_model.reset();
        different_physics = false;
        return live;
    }

    template<class Record>
    double evaluate(const Snapshot& snapshot, const float* inputs, std::size_t count,
                    std::size_t ticks, int width, RolloutStatus* status, bool matched_position, const Record& record) {
        std::lock_guard batch_lock(batch_mutex);
        std::optional<Snapshot> imported;
        if (snapshot.model != model) {
            if (snapshot.model != execution_model)
                throw std::invalid_argument("snapshot belongs to a different world");
            imported = snapshot;
            imported->model = model;
            auto clear = [&](int field) {
                const int offset = mj_stateSize(model.get(), mjSTATE_INTEGRATION & (field - 1));
                auto first = imported->integration.begin() + offset;
                auto last = first + mj_stateSize(model.get(), field);
                const bool active = std::any_of(first, last, [](double x) { return x != 0.; });
                std::fill(first, last, 0.);
                return active;
            };
            // The previous acceleration is a numerical solver seed, not observed physics.
            // Preserve it for exact clones, discard it across different dynamics.
            bool hidden_load = false;
            if (!copy_applied_load) {
                const bool generalized = clear(mjSTATE_QFRC_APPLIED);
                hidden_load = clear(mjSTATE_XFRC_APPLIED) || generalized;
            }
            if (different_physics || hidden_load) clear(mjSTATE_WARMSTART);
            imported->contact = {};  // interval measurement, rebuilt on the first step
        }
        const Snapshot& branch = imported ? *imported : snapshot;
        if (!count || !ticks || !World::supports_width(width)) {
            throw std::invalid_argument("expected nonempty world-target or policy-command schedules");
        }
        const auto lanes = std::min(worlds.size(), count);
        const auto simulate = [&](std::size_t lane) {
            auto& world = *worlds[lane];
            for (auto i = lane; i < count; i += lanes) {
                world.restore(branch);
                auto& result = status[i];
                result = {};
                try {
                    for (; result.steps < ticks; ++result.steps) {
                        const auto row = i*ticks + result.steps;
                        Command command = world.command_at(inputs + row*width, width, matched_position);
                        world.step(command, &result.time);
                        record(row, world, command);
                    }
                } catch (const TrajectoryFailure& e) {
                    result.failure = e.what();
                } catch (const std::invalid_argument& e) {
                    result.failure = e.what();
                }
            }
        };
        if (lanes == 1) {
            simulate(0);
            return 0;
        }

        const auto start = StepTimes::Clock::now();
        std::exception_ptr error;
        std::chrono::duration<double> longest{};
        const std::function<void(std::size_t)> job = [&](std::size_t lane) {
            const auto lane_start = StepTimes::Clock::now();
            std::exception_ptr failure;
            try {
                simulate(lane);
            } catch (...) {
                failure = std::current_exception();
            }
            const std::chrono::duration<double> elapsed = StepTimes::Clock::now() - lane_start;
            std::lock_guard lock(work_mutex);
            if (failure) error = failure;
            longest = std::max(longest, elapsed);
        };
        {
            std::lock_guard lock(work_mutex);
            work = &job;
            active = lanes;
            pending = lanes - 1;
            ++generation;
        }
        ready.notify_all();
        job(0);
        std::unique_lock lock(work_mutex);
        finished.wait(lock, [&] { return pending == 0; });
        work = nullptr;
        if (error) std::rethrow_exception(error);
        const std::chrono::duration<double> elapsed = StepTimes::Clock::now() - start;
        return (elapsed - longest).count();
    }

private:
    void serve(std::size_t lane) {
        std::size_t seen = 0;
        std::unique_lock lock(work_mutex);
        for (;;) {
            ready.wait(lock, [&] { return stopping || generation != seen; });
            if (stopping) return;
            seen = generation;
            if (!work || lane >= active) continue;
            const auto* job = work;
            lock.unlock();
            (*job)(lane);
            lock.lock();
            if (--pending == 0) finished.notify_one();
        }
    }

    void shutdown() {
        {
            std::lock_guard lock(work_mutex);
            stopping = true;
        }
        ready.notify_all();
        for (auto& thread : threads) thread.join();
    }

    ModelPtr model, execution_model;
    bool different_physics = false, copy_applied_load = false;
    std::vector<std::unique_ptr<World>> worlds;
    std::vector<std::thread> threads;
    std::mutex batch_mutex, work_mutex;
    std::condition_variable ready, finished;
    const std::function<void(std::size_t)>* work = nullptr;
    std::size_t generation = 0, active = 0, pending = 0;
    bool stopping = false;
};

}  // namespace unifp
