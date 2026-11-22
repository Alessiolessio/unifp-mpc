"""Branch from the same settled approach to a fixed vertical box face."""
import json
import tempfile
from pathlib import Path

import mujoco
import numpy as np

from force_box import POLICY, ROOT
from prepare_scene import scene_template
from run_world import command_schedule, spherical
from unifp_native import World


def main():
    model = scene_template(fixed_face_x=.89)
    assert model.body("box_body").jntnum.item() == 0
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("spawn").id)
    mujoco.mj_forward(model, data)
    surface = model.geom("box_collision").id
    assert all(surface not in c.geom for c in data.contact)
    with tempfile.TemporaryDirectory() as tmp:
        scene = str(Path(tmp) / "fixed.mjb")
        mujoco.mj_saveModel(model, scene)
        world = World(scene, str(POLICY))
    warm = world.run(command_schedule("push")[:50])
    assert not warm["other_robot_box_impulse_Ns"].any()
    nominal = np.array([.42, 0., .01])
    approach_cmd = np.zeros((100, 12), np.float32)
    approach_cmd[:, 3:6] = spherical(np.linspace([.20, 0, .10], nominal, 100))
    approach = world.run(approach_cmd)
    snap = world.snapshot()
    rows = []
    for newtons in (0., 10., 20.):
        for matched in ((False, True) if newtons else (False,)):
            world.restore(snap)
            schedule = np.zeros((150, 12), np.float32)
            force = newtons*np.minimum(1., np.arange(150)/50)
            target = np.tile(nominal, (150, 1))
            if matched:
                target[:, 0] += force / 200.
            else:
                schedule[:, 6] = force
            schedule[:, 3:6] = spherical(target)
            assert schedule[:, 3].max() < .55
            t = world.run(schedule)
            assert not t["other_robot_box_impulse_Ns"].any()
            window = slice(-75, None)
            normal = t["gripper_on_box_N"][window, 0]
            face_normals = []
            for pose in t["qpos"][window][t["contact_duty"][window] > 0]:
                data.qpos[:] = pose
                mujoco.mj_forward(model, data)
                face_normals.extend(abs(c.frame[0]) for c in data.contact if surface in c.geom)
            assert min(face_normals) > .99
            cmd = np.linalg.norm(t["policy_command"][window, 6:9], axis=1)
            rows.append(dict(request_N=newtons, matched=matched, issued_mean_N=float(cmd.mean()),
                             mean_normal_N=float(normal.mean()), peak_normal_N=float(normal.max()),
                             impulse_Ns=float(normal.sum()*.02),
                             duty=float(t["contact_duty"][window].mean()),
                             finger_dx_mm=float(1000*(t["ee_world"][window, 0].mean()
                                                      - approach["ee_world"][-1, 0]))))
    print(json.dumps(rows, indent=2))
    ROOT.mkdir(exist_ok=True, parents=True)
    (ROOT / "fixed.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
