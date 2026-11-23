"""Paired standing response to a finite physical downward wrench at the finger COM."""
import argparse
import csv
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np

from fixed_policy_loads import DT, MODES, N, OUT, check_workspace, command, ee_goal, first, rms
from native_probe import set_finger_load_z
from unifp_native import World


def impulse_check(model):
    model.key("spawn").qpos[2] += 1
    with TemporaryDirectory() as tmp:
        scene = str(Path(tmp)/"air.mjb")
        mujoco.mj_saveModel(model, scene)
        w = World(scene, "artifacts/encoder_actor.onnx")
        snap = w.snapshot()
        momentum = []
        data = mujoco.MjData(model)
        for f in (0., -10.):
            w.restore(snap)
            set_finger_load_z(w, f)
            t = w.run(command([.35, 0., .10])[None])
            data.qpos[:] = t["qpos"][0]
            data.qvel[:] = t["qvel"][0]
            mujoco.mj_forward(model, data)
            mujoco.mj_subtreeVel(model, data)
            assert data.ncon == 0
            momentum.append(model.body_subtreemass[0]*data.subtree_linvel[0].copy())
        diff = momentum[1]-momentum[0]
        np.testing.assert_allclose(diff, [0, 0, -.2], atol=.004)
        print("Added free-flight impulse in N s", diff)
    model.key("spawn").qpos[2] -= 1


def run(case, m, folder):
    rng = np.random.default_rng(case)
    xyz = np.array([rng.uniform(.30, .42), rng.uniform(-.05, .05), rng.uniform(.03, .12)])
    f = rng.uniform(-12., -6.)
    a = command(xyz)
    b, c = a.copy(), command(xyz+[0., 0., .01])
    b[8] = 2.
    w = World("artifacts/spot.mjb", "artifacts/encoder_actor.onnx")
    w.run(np.tile(a, (100, 1)))
    snap = w.snapshot()
    np.testing.assert_allclose(ee_goal(w.base_pose[None], b), ee_goal(w.base_pose[None], c),
                               rtol=0, atol=1e-6)
    joint_q = [m.joint(name).qposadr.item() for name in
               ("arm_sh0", "arm_sh1", "arm_el0", "arm_el1", "arm_wr0", "arm_wr1")]
    rows = []
    for mode, cmd in zip(MODES, (a, b, c), strict=True):
        check_workspace(cmd)
        w.restore(snap)
        u = np.tile(cmd, (N, 1))
        free = w.run(u)
        w.restore(snap)
        before = w.run(u[:40])
        np.testing.assert_array_equal(before["qpos"], free["qpos"][:40])
        set_finger_load_z(w, f)
        loaded = w.run(u[40:60])
        set_finger_load_z(w, 0.)
        after = w.run(u[60:])
        z = np.concatenate([t["ee_world"][:, 2] for t in (before, loaded, after)])
        pose = np.concatenate([t["qpos"] for t in (before, loaded, after)])
        delta = z - free["ee_world"][:, 2]
        tilt = np.rad2deg(np.arccos(np.clip(1-2*np.sum(pose[:, 4:6]**2, axis=1), -1, 1)))
        rows.append(dict(case=case, mode=mode, physical_fz_N=f, peak_drop_m=-delta[40:60].min(),
                         loaded_rms_m=rms(delta[40:60]), recovery_s=first(abs(delta[60:]) < .005, 10),
                         arm_peak_delta_rad=abs((pose-free["qpos"])[40:60, joint_q]).max(),
                         nominal_z_rms_m=rms(z[40:60]-ee_goal(pose[40:60], a)[:, 2]),
                         max_tilt_deg=tilt.max()))
        np.savez_compressed(folder/f"pulse_{case}_{mode}.npz", z=z, unperturbed=free["ee_world"][:, 2])
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("split", choices=("development", "held"))
    args = p.parse_args()
    folder = OUT/args.split
    folder.mkdir(parents=True, exist_ok=True)
    m = mujoco.MjModel.from_binary_path("artifacts/spot.mjb")
    impulse_check(m)
    rows = []
    for case in (range(2) if args.split == "development" else range(100, 112)):
        rows.extend(run(case, m, folder))
    with (folder/"pulse.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
