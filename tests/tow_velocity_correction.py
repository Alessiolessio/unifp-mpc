"""Test one prespecified +0.016 m/s ordinary correction against the original loaded target."""
import csv
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from box_mpc import yaw_of
from fixed_policy_loads import (N, OUT, body_velocity, command, first, length, prepare_case, rms)
from unifp_native import World


def run(seed, scene):
    _, warm, xyz, velocity, _ = prepare_case("tether", seed, False, scene, None)
    live = World(str(scene), "artifacts/encoder_actor.onnx")
    pre = live.run(warm)
    direction = velocity/length(velocity)
    target = velocity-.08*direction
    corrected = command(xyz, target+.016*direction)
    result = live.run(np.tile(corrected, (N,1)))
    v = body_velocity(result["qpos"], result["qvel"][:,:3])
    with np.load(OUT/"held"/f"tether_{seed}_C-16.npz") as c:
        np.testing.assert_array_equal(c["start"], pre["qpos"][-1])
        np.testing.assert_allclose(c["effective_v"], target, atol=3e-8)
    # Use the same original task target for all three errors.
    out = []
    for mode in ("B-16", "C-16", "corrected"):
        if mode == "corrected":
            pos = result["qpos"][:, -7:-5] - pre["qpos"][-1,-7:-5]
            yaw = yaw_of(pre["qpos"][-1,3:7])
            along = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]) @ direction
            progress = pos @ along
            speed = v
        else:
            with np.load(OUT/"held"/f"tether_{seed}_{mode}.npz") as f:
                speed, progress = f["body_velocity"], f["box_progress"]
        out.append(dict(seed=seed, treatment=mode, tracking_rms_m_s=rms(length(speed-target)),
                        mean_speed_m_s=np.mean(speed@direction), progress_m=progress[-1],
                        time_25cm_s=first(progress >= .25)))
    return out


if __name__ == "__main__":
    with TemporaryDirectory() as tmp:
        rows = [r for i in range(100,112) for r in run(i,Path(tmp)/"scene.mjb")]
    with Path("artifacts/cone_force/tow.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=rows[0], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
