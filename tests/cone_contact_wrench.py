"""Replay recorded actuator targets and integrate hand contact wrenches at 200 Hz."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import mujoco
import numpy as np

from native_probe import DriveProbe


def analyze(folder):
    path = str(folder / "scene.mjb")
    model = mujoco.MjModel.from_binary_path(path)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("spawn").id)
    mujoco.mj_forward(model, data)
    drive = DriveProbe(path)
    motors = np.argsort(model.actuator_user[:, 0])
    nominal = model.actuator_user[motors, 1].astype(np.float32)
    ji = model.actuator_trnid[motors, 0]
    cone = model.body("traffic_cone").id
    hand = [model.body(n).id for n in ("arm_link_fngr", "arm_link_wr1")]
    e = np.load(folder / "episode.npz")
    q, force = e["qpos"], e["gripper_on_cone_N"]
    per_geom = defaultdict(float)
    axial, axial_normals, weights, moments = [], [], [], []
    wrench = np.zeros(6)
    for i, action in enumerate(e["action"]):
        target = nominal + 0.25 * action
        net = np.zeros(3)
        for _ in range(4):
            data.ctrl[motors] = drive.torque(
                data.qpos[model.jnt_qposadr[ji]], data.qvel[model.jnt_dofadr[ji]], target
            )
            mujoco.mj_step(model, data)
            axis = data.xmat[cone].reshape(3, 3)[:, 2]
            # Positive scalar increases the upward alignment of this instantaneous cone axis.
            lift = np.cross(axis, [0.0, 0.0, 1.0])
            for j, c in enumerate(data.contact):
                a, b = model.geom_bodyid[c.geom]
                if not ((a == cone and b in hand) or (b == cone and a in hand)):
                    continue
                mujoco.mj_contactForce(model, data, j, wrench)
                frame = c.frame.reshape(3, 3)
                sign = 1 if b == cone else -1
                f = sign * (wrench[:3] @ frame)
                net += f / 4
                impulse = np.linalg.norm(f) * 0.005
                if impulse == 0:
                    continue
                torque = np.cross(c.pos - data.xipos[cone], f) + sign * (wrench[3:] @ frame)
                axial.append((c.pos - data.xpos[cone]) @ axis)
                weights.append(impulse)
                axial_normals.append(sign * (frame[0] @ axis))
                moments.append((torque @ lift) * 0.005)
                per_geom[model.geom(c.geom[1 if b == cone else 0]).name] += impulse
        np.testing.assert_array_equal(q[i], data.qpos)
        np.testing.assert_allclose(force[i], net, atol=1e-10)
        mujoco.mj_kinematics(model, data)
    return dict(
        geom_Ns=dict(per_geom),
        impulse_weighted_axial_m=np.average(axial, weights=weights),
        normal_dot_axis=np.average(axial_normals, weights=weights),
        positive_tipping_angular_impulse_Nms=np.maximum(moments, 0.0).sum(),
        negative_tipping_angular_impulse_Nms=np.minimum(moments, 0.0).sum(),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("episodes", type=Path, nargs="+")
    print(json.dumps({str(f): analyze(f) for f in p.parse_args().episodes}, indent=2))
