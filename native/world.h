#pragma once

#include "plant.h"

#include <cstdint>
#include <cstring>

#include <core/session/onnxruntime_cxx_api.h>

namespace unifp {

using Command = std::array<float, 12>;

class World {
public:
    World(const std::string& scene, const std::string& policy)
        : plant(scene), env(ORT_LOGGING_LEVEL_WARNING, "unifp"),
          session(env, policy.c_str(), options()),
          memory(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
          input(Ort::Value::CreateTensor<float>(memory, history.data(), history.size(),
                                                input_shape.data(), 2)),
          output(Ort::Value::CreateTensor<float>(memory, action.data(), action.size(),
                                                 output_shape.data(), 2)) {
        const mjModel* m = plant.model.get();
        const int base = mj_name2id(m, mjOBJ_JOINT, "base");
        finger = mj_name2id(m, mjOBJ_BODY, "arm_link_fngr");
        wrist = mj_name2id(m, mjOBJ_BODY, "arm_link_wr1");
        box = mj_name2id(m, mjOBJ_BODY, "box_body");
        if (base < 0 || m->jnt_type[base] != mjJNT_FREE || finger < 0 || wrist < 0) {
            throw std::invalid_argument("expected floating Spot with arm");
        }
        base_q = m->jnt_qposadr[base];
        base_v = m->jnt_dofadr[base];
    }

    World(const World&) = delete;
    World& operator=(const World&) = delete;

    // The input is the new command for this tick. Contact statistics cover the four
    // following physics steps; kinematics on return are at the end of that interval.
    void step(const Command& command) {
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
        session.Run(run_options, &in_name, &input, 1, &out_name, &output, 1);
        JointVector target;
        for (int i = 0; i < joints; ++i) {
            if (!std::isfinite(action[i])) throw std::runtime_error("non-finite policy action");
            action[i] = std::clamp(action[i], -100.f, 100.f);
            target[i] = plant.drive[i].nominal + .25f * action[i];
        }
        gripper_force.fill(0);
        ground_force.fill(0);
        peak_forward = contact_duty = other_impulse = 0;
        for (int substep = 0; substep < substeps; ++substep) {
            plant.advance(target);
            if (box >= 0) accumulate_contact();
        }
        ++tick;
        if (std::abs(plant.data->time - tick * policy_dt) > 1e-6) {
            throw std::runtime_error("MuJoCo integration reset");
        }
        mj_kinematics(plant.model.get(), plant.data.get());
        const auto tilt = roll_pitch();
        if (std::max(std::abs(tilt[0]), std::abs(tilt[1])) > 1.04719755) {
            throw std::runtime_error("base tilt exceeded 60 degrees at tick " + std::to_string(tick));
        }
    }

    Plant plant;
    JointVector action{};
    std::array<double, 3> gripper_force{}, ground_force{};
    double peak_forward = 0, contact_duty = 0, other_impulse = 0;
    int finger = -1;

private:
    static Ort::SessionOptions options() {
        Ort::SessionOptions s;
        s.SetIntraOpNumThreads(1);
        s.SetInterOpNumThreads(1);
        return s;
    }

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
            if (a != box && b != box) continue;
            const int other = a == box ? b : a;
            mjtNum wrench[6], world[3];
            mj_contactForce(m, d, i, wrench);
            mju_mulMatTVec3(world, contact.frame, wrench);
            // Contact-frame rows are world axes. The reported wrench acts on geom[1].
            const double sign = b == box ? 1 : -1;
            if (other == 0) {
                for (int k = 0; k < 3; ++k) ground_force[k] += sign * world[k] / substeps;
            } else if (other == finger || other == wrist) {
                for (int k = 0; k < 3; ++k) hand[k] += sign * world[k];
            } else {
                other_impulse += mju_norm3(world) * m->opt.timestep;
            }
        }
        for (int k = 0; k < 3; ++k) gripper_force[k] += hand[k] / substeps;
        peak_forward = std::max(peak_forward, hand[0]);
        if (mju_norm3(hand.data()) > 1) contact_duty += 1.0 / substeps;
    }

    int base_q, base_v, wrist, box, tick = 0;
    std::array<float, 32 * 79> history{};
    const std::array<int64_t, 2> input_shape{1, 2528}, output_shape{1, 19};
    Ort::Env env;
    Ort::Session session;
    Ort::MemoryInfo memory;
    Ort::Value input, output;
    Ort::RunOptions run_options;
};

}  // namespace unifp
