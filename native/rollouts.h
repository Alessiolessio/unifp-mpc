#pragma once

#include "world.h"

#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
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
        return live;
    }

    template<class Record>
    double evaluate(const Snapshot& snapshot, const float* inputs, std::size_t count,
                    std::size_t ticks, int width, RolloutStatus* status, const Record& record) {
        std::lock_guard batch_lock(batch_mutex);
        if (snapshot.model != model) throw std::invalid_argument("snapshot belongs to a different world");
        if (!count || !ticks || !World::supports_width(width)) {
            throw std::invalid_argument("expected nonempty world-target or policy-command schedules");
        }
        const auto lanes = std::min(worlds.size(), count);
        const auto simulate = [&](std::size_t lane) {
            auto& world = *worlds[lane];
            for (auto i = lane; i < count; i += lanes) {
                world.restore(snapshot);
                auto& result = status[i];
                result = {};
                try {
                    for (; result.steps < ticks; ++result.steps) {
                        const auto row = i*ticks + result.steps;
                        Command command = world.command_at(inputs + row*width, width);
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

    ModelPtr model;
    std::vector<std::unique_ptr<World>> worlds;
    std::vector<std::thread> threads;
    std::mutex batch_mutex, work_mutex;
    std::condition_variable ready, finished;
    const std::function<void(std::size_t)>* work = nullptr;
    std::size_t generation = 0, active = 0, pending = 0;
    bool stopping = false;
};

}  // namespace unifp
