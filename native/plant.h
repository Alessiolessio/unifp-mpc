#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>

namespace unifp {

constexpr int joints = 19;
constexpr double physics_dt = .005;
constexpr int substeps = 4;
constexpr double policy_dt = substeps * physics_dt;
using JointVector = std::array<float, joints>;
using ModelPtr = std::shared_ptr<const mjModel>;
using DataPtr = std::unique_ptr<mjData, decltype(&mj_deleteData)>;

inline ModelPtr load_model(const std::string& path) {
    auto* model = mj_loadModel(path.c_str(), nullptr);
    if (!model) throw std::invalid_argument("cannot load UniFP MJB: " + path);
    return ModelPtr(model, mj_deleteModel);
}

struct TargetDelay {
    std::array<JointVector, 4> targets{};
    int write = 0;
};

struct Drive {
    int motor, qpos, dof, lag;
    float nominal, kp, kd, limit;
};

class Plant {
public:
    explicit Plant(const std::string& path) : Plant(load_model(path)) {}

    explicit Plant(ModelPtr shared_model)
        : model(std::move(shared_model)), data(mj_makeData(model.get()), mj_deleteData) {
        if (!data || model->nu != joints || model->nuser_actuator != 6 ||
            std::abs(model->opt.timestep - physics_dt) > 1e-12) {
            throw std::invalid_argument("expected a prepared 5 ms UniFP Spot MJB");
        }
        const int key = mj_name2id(model.get(), mjOBJ_KEY, "spawn");
        const int table = mj_name2id(model.get(), mjOBJ_NUMERIC, "knee_angle_limit");
        if (key < 0 || table < 0 || model->numeric_size[table] < 4 ||
            model->numeric_size[table] % 2 != 0) {
            throw std::invalid_argument("missing spawn pose or knee torque table");
        }
        const mjtNum* entries = model->numeric_data + model->numeric_adr[table];
        for (int i = 0; i < model->numeric_size[table]; i += 2) {
            angles.push_back(static_cast<float>(entries[i]));
            limits.push_back(static_cast<float>(entries[i + 1]));
        }
        mj_resetDataKeyframe(model.get(), data.get(), key);
        mj_forward(model.get(), data.get());
        std::array<bool, joints> seen{};
        for (int motor = 0; motor < joints; ++motor) {
            const mjtNum* meta = model->actuator_user + motor * 6;
            const int i = static_cast<int>(meta[0]);
            if (i < 0 || i >= joints || seen[i] || meta[5] < 0 || meta[5] > 3 ||
                model->actuator_trntype[motor] != mjTRN_JOINT) {
                throw std::invalid_argument("invalid UniFP motor metadata");
            }
            seen[i] = true;
            const int joint = model->actuator_trnid[2 * motor];
            drive[i] = {motor, model->jnt_qposadr[joint], model->jnt_dofadr[joint],
                        static_cast<int>(meta[5]), static_cast<float>(meta[1]),
                        static_cast<float>(meta[2]), static_cast<float>(meta[3]),
                        static_cast<float>(meta[4])};
            // Reset writes the spawn target once. Unfilled delays read that oldest target.
            for (auto& frame : delay.targets) frame[i] = static_cast<float>(data->qpos[drive[i].qpos]);
        }
    }

    void apply(const JointVector& target) {
        delay.targets[delay.write] = target;
        for (int i = 0; i < joints; ++i) {
            const Drive& d = drive[i];
            const float q = static_cast<float>(data->qpos[d.qpos]);
            const float dq = static_cast<float>(data->qvel[d.dof]);
            const float delayed = delay.targets[(delay.write + 4 - d.lag) % 4][i];
            const float torque = d.kp * (delayed - q) - d.kd * dq;
            const float bound = d.limit == 0 ? knee_limit(q) : d.limit;
            data->ctrl[d.motor] = std::clamp(torque, -bound, bound);
        }
        delay.write = (delay.write + 1) % 4;
    }

    void advance(const JointVector& target) {
        apply(target);
        mj_step(model.get(), data.get());
    }

    ModelPtr model;
    DataPtr data;
    std::array<Drive, joints> drive{};
    TargetDelay delay;

private:
    float knee_limit(float q) const {
        const auto upper = std::lower_bound(angles.begin(), angles.end(), q);
        if (upper == angles.begin()) return limits.front();
        if (upper == angles.end()) return limits.back();
        const auto i = upper - angles.begin();
        const float f = (q - angles[i - 1]) / (angles[i] - angles[i - 1]);
        return limits[i - 1] + f * (limits[i] - limits[i - 1]);
    }

    std::vector<float> angles, limits;
};

}  // namespace unifp
