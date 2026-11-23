#pragma once

#include "../native/plant.h"
#include <pybind11/stl.h>

class DriveProbe : public unifp::Plant {
public:
    using Plant::Plant;

    unifp::JointVector torque(const unifp::JointVector& q, const unifp::JointVector& dq,
                              const unifp::JointVector& target) {
        for (int i = 0; i < unifp::joints; ++i) {
            data->qpos[drive[i].qpos] = q[i];
            data->qvel[drive[i].dof] = dq[i];
        }
        apply(target);
        return last_effort();
    }

    unifp::JointVector last_effort() const {
        unifp::JointVector effort;
        for (int i = 0; i < unifp::joints; ++i) effort[i] = data->ctrl[drive[i].motor];
        return effort;
    }

    unifp::JointVector step(const unifp::JointVector& target) {
        advance(target);
        unifp::JointVector q;
        for (int i = 0; i < unifp::joints; ++i) q[i] = data->qpos[drive[i].qpos];
        return q;
    }
};
