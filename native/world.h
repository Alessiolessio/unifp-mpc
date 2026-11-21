#pragma once

#include "plant.h"

#include <chrono>
#include <cstdint>
#include <cstring>

#include <core/session/onnxruntime_cxx_api.h>

namespace unifp {

using Command = std::array<float, 12>;
using History = std::array<float, 32 * 79>;

struct Policy {
    explicit Policy(const std::string& path) : session(env, path.c_str(), options()) {}

    static Ort::SessionOptions options() {
        Ort::SessionOptions s;
        s.SetIntraOpNumThreads(1);
        s.SetInterOpNumThreads(1);
        return s;
    }

    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "unifp"};
    Ort::Session session;
    Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
};

struct ContactStats {
    std::array<double, 3> gripper_force{}, ground_force{};
    double peak_forward = 0, contact_duty = 0, other_impulse = 0;
    bool gripper_touch = false;
};

struct Snapshot {
    ModelPtr model;
    std::vector<mjtNum> integration;
    History history;
    JointVector action;
    TargetDelay delay;
    ContactStats contact;
    int tick;
    std::array<mjWarningStat, mjNWARNING> warnings{};
};

struct StepTimes {
    using Clock = std::chrono::steady_clock;
    std::chrono::duration<double> inference{}, physics{};
};

struct TrajectoryFailure : std::runtime_error {
    using std::runtime_error::runtime_error;
};

class World {
public:
    World(const std::string& scene, const std::string& policy)
        : World(load_model(scene), std::make_shared<Policy>(policy)) {}

    World(ModelPtr model, std::shared_ptr<Policy> shared_policy)
        : plant(std::move(model)), policy(std::move(shared_policy)),
          input(Ort::Value::CreateTensor<float>(policy->memory, history.data(), history.size(),
                                                input_shape.data(), 2)),
          output(Ort::Value::CreateTensor<float>(policy->memory, action.data(), action.size(),
                                                 output_shape.data(), 2)) {
        const mjModel* m = plant.model.get();
        const int base = mj_name2id(m, mjOBJ_JOINT, "base");
        finger = mj_name2id(m, mjOBJ_BODY, "arm_link_fngr");
        wrist = mj_name2id(m, mjOBJ_BODY, "arm_link_wr1");
        object_body = mj_name2id(m, mjOBJ_BODY, "box_body");
        if (object_body < 0) object_body = mj_name2id(m, mjOBJ_BODY, "traffic_cone");
        if (base < 0 || m->jnt_type[base] != mjJNT_FREE || finger < 0 || wrist < 0) {
            throw std::invalid_argument("expected floating Spot with arm");
        }
        base_q = m->jnt_qposadr[base];
        base_v = m->jnt_dofadr[base];
    }

    World(const World&) = delete;
    World& operator=(const World&) = delete;

    Snapshot snapshot() const {
        const auto* m = plant.model.get();
        Snapshot state{plant.model, std::vector<mjtNum>(mj_stateSize(m, mjSTATE_INTEGRATION)),
                       history, action, plant.delay, last_contact, tick};
        mj_getState(m, plant.data.get(), state.integration.data(), mjSTATE_INTEGRATION);
        std::copy_n(plant.data->warning, mjNWARNING, state.warnings.begin());
        return state;
    }

    void restore(const Snapshot& state) {
        if (state.model != plant.model) {
            throw std::invalid_argument("snapshot belongs to a different world");
        }
        mj_setState(plant.model.get(), plant.data.get(), state.integration.data(), mjSTATE_INTEGRATION);
        std::copy_n(state.warnings.begin(), mjNWARNING, plant.data->warning);
        history = state.history;
        action = state.action;
        plant.delay = state.delay;
        last_contact = state.contact;
        tick = state.tick;
        // The next observation reads qpos/qvel directly. mj_step rebuilds contacts and dynamics.
        mj_kinematics(plant.model.get(), plant.data.get());
    }

    static bool supports_width(std::size_t width) {
        return (width >= 3 && width <= 6) || width == 12;
    }

    Command command_at(const float* sample, int width) const {
        Command command{};
        if (width == 12) {
            std::copy_n(sample, 12, command.begin());
            return command;
        }
        // World XYZ, optionally followed by body vx, vy (m/s) and yaw rate (rad/s).
        if (width >= 4) command[0] = sample[3];
        if (width >= 5) command[1] = sample[4];
        if (width == 6) command[2] = sample[5];
        // Project at this candidate's pre-action root pose.
        if (!std::all_of(sample, sample + 3, [](float x) { return std::isfinite(x); })) {
            throw std::invalid_argument("non-finite world target");
        }
        const mjtNum* root = plant.data->qpos + base_q;
        const float w = root[3], x = root[4], y = root[5], z = root[6];
        const float yaw = std::atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z));
        const float c = std::cos(yaw), s = std::sin(yaw);
        const float dx = sample[0] - static_cast<float>(root[0]);
        const float dy = sample[1] - static_cast<float>(root[1]);
        const float rx = c*dx + s*dy - .35f, ry = -s*dx + c*dy;
        const float rz = sample[2] - static_cast<float>(root[2]) - .12f;
        const float radius = std::sqrt(rx*rx + ry*ry + rz*rz);
        constexpr float rad = 3.14159265f / 180;
        command[3] = std::clamp(radius, .15f, .55f);
        command[4] = std::clamp(std::asin(std::clamp(rz / (radius + 1e-8f), -1.f, 1.f)),
                                -40 * rad, 50 * rad);
        command[5] = std::clamp(std::atan2(ry, rx), -90 * rad, 90 * rad);
        return command;
    }

    // The input is the new command for this tick. Contact statistics cover the four
    // following physics steps; kinematics on return are at the end of that interval.
    void step(const Command& command, StepTimes* times = nullptr) {
        if (!std::all_of(command.begin(), command.end(), [](float x) { return std::isfinite(x); })) {
            throw std::invalid_argument("non-finite UniFP command");
        }
        std::memmove(history.data(), history.data() + 79, (history.size() - 79) * sizeof(float));
        float* frame = history.data() + history.size() - 79;
        const auto rp = roll_pitch();
        frame[0] = rp[0];
        frame[1] = rp[1];
        for (int k = 0; k < 3; ++k) {
            frame[2 + k] = .25f * static_cast<float>(plant.data->qvel[base_v + 3 + k]);
        }
        for (int i = 0; i < joints; ++i) {
            const Drive& d = plant.drive[i];
            frame[5 + i] = static_cast<float>(plant.data->qpos[d.qpos]) - d.nominal;
            frame[24 + i] = .05f * static_cast<float>(plant.data->qvel[d.dof]);
            frame[43 + i] = action[i];
        }
        const float phase = static_cast<float>(tick) * static_cast<float>(policy_dt) / .64f
                            * 2 * 3.14159265f;
        frame[62] = std::sin(phase);
        frame[63] = std::cos(phase);
        frame[64] = 2 * command[0];
        frame[65] = 2 * command[1];
        frame[66] = .25f * command[2];
        std::copy_n(command.begin() + 3, 3, frame + 67);
        std::fill_n(frame + 70, 3, 0.f);
        for (int i = 0; i < 6; ++i) frame[73 + i] = .01f * command[6 + i];
        for (int i = 0; i < 79; ++i) frame[i] = std::clamp(frame[i], -100.f, 100.f);
        const char* in_name = "history";
        const char* out_name = "action";
        StepTimes::Clock::time_point start;
        if (times) start = StepTimes::Clock::now();
        policy->session.Run(run_options, &in_name, &input, 1, &out_name, &output, 1);
        if (times) times->inference += StepTimes::Clock::now() - start;
        JointVector target;
        for (int i = 0; i < joints; ++i) {
            if (!std::isfinite(action[i])) throw TrajectoryFailure("non-finite policy action");
            action[i] = std::clamp(action[i], -100.f, 100.f);
            target[i] = plant.drive[i].nominal + .25f * action[i];
        }
        last_contact = {};
        if (times) start = StepTimes::Clock::now();
        for (int substep = 0; substep < substeps; ++substep) {
            plant.advance(target);
            if (object_body >= 0) accumulate_contact();
        }
        ++tick;
        if (std::abs(plant.data->time - tick * policy_dt) > 1e-6) {
            throw TrajectoryFailure("MuJoCo integration reset");
        }
        for (int i = 0; i < mjNWARNING; ++i) {
            const auto& warning = plant.data->warning[i];
            if (i != mjWARN_VGEOMFULL && warning.number) {
                throw TrajectoryFailure(mju_warningText(i, warning.lastinfo));
            }
        }
        const auto finite = [](mjtNum x) { return std::isfinite(x); };
        if (!std::all_of(plant.data->qpos, plant.data->qpos + plant.model->nq, finite) ||
            !std::all_of(plant.data->qvel, plant.data->qvel + plant.model->nv, finite)) {
            throw TrajectoryFailure("non-finite integrated state");
        }
        mj_kinematics(plant.model.get(), plant.data.get());
        if (times) times->physics += StepTimes::Clock::now() - start;
        const auto tilt = roll_pitch();
        if (std::max(std::abs(tilt[0]), std::abs(tilt[1])) > 1.04719755) {
            throw TrajectoryFailure("base tilt exceeded 60 degrees at tick " + std::to_string(tick));
        }
    }

    Plant plant;
    std::shared_ptr<Policy> policy;
    JointVector action{};
    ContactStats last_contact;
    int finger = -1;

private:
    std::array<double, 2> roll_pitch() const {
        const mjtNum* q = plant.data->qpos + base_q + 3;
        const auto w = q[0], x = q[1], y = q[2], z = q[3];
        return {std::atan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y)),
                std::asin(std::clamp(2 * (w*y - z*x), -1., 1.))};
    }

    void accumulate_contact() {
        const auto* m = plant.model.get();
        const auto* d = plant.data.get();
        std::array<double, 3> hand{};
        for (int i = 0; i < d->ncon; ++i) {
            const mjContact& contact = d->contact[i];
            const int a = m->geom_bodyid[contact.geom[0]], b = m->geom_bodyid[contact.geom[1]];
            if (a != object_body && b != object_body) continue;
            const int other = a == object_body ? b : a;
            mjtNum wrench[6], world[3];
            mj_contactForce(m, d, i, wrench);
            mju_mulMatTVec3(world, contact.frame, wrench);
            // Contact-frame rows are world axes. The reported wrench acts on geom[1].
            const double sign = b == object_body ? 1 : -1;
            if (other == 0) {
                for (int k = 0; k < 3; ++k) last_contact.ground_force[k] += sign * world[k] / substeps;
            } else if (other == finger || other == wrist) {
                if (contact.efc_address >= 0) last_contact.gripper_touch = true;
                for (int k = 0; k < 3; ++k) hand[k] += sign * world[k];
            } else {
                last_contact.other_impulse += mju_norm3(world) * m->opt.timestep;
            }
        }
        for (int k = 0; k < 3; ++k) last_contact.gripper_force[k] += hand[k] / substeps;
        last_contact.peak_forward = std::max(last_contact.peak_forward, hand[0]);
        if (mju_norm3(hand.data()) > 1) last_contact.contact_duty += 1.0 / substeps;
    }

    int base_q, base_v, wrist, object_body, tick = 0;
    History history{};
    const std::array<int64_t, 2> input_shape{1, 2528}, output_shape{1, 19};
    Ort::Value input, output;
    Ort::RunOptions run_options;
};

}  // namespace unifp
