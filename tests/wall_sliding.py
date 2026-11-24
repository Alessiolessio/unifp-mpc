"""Compare the established compliant-wall walks with a heading/slide cost at the same budget."""
import csv
import json
from pathlib import Path

import numpy as np

from kinematics import yaw_of
from wall_mpc import POLICY, Rollouts, World, episode


def body_frame(v, root):
    xyz = -root[..., 4:7]
    return v + 2*np.cross(xyz, np.cross(xyz, v) + root[..., 3:4]*v)


def measure(folder, s):
    with np.load(folder/"episode.npz") as e:
        q, v, f = e["qpos"], e["qvel"], e["ee_world"]
        q0 = e["initial_qpos"] if "initial_qpos" in e else e["warm_qpos"][-1]
        yaw = yaw_of(q[:, 3:7])
        yaw0 = yaw_of(q0[3:7])
        heading = np.unwrap(np.r_[yaw0, yaw])[1:] - yaw0
        loaded = e["wall_normal_N"].mean(axis=1) > 1
        dy = s["finger_delta_m"][1]
        f0 = f[-1] - s["finger_delta_m"]
        d = np.diff(np.vstack((f0, f)), axis=0)
        local_hand = body_frame(f-q[:, :3], q)
        hand0 = body_frame(f0-q0[:3], q0)
        body_vy = body_frame(v[:, :3], q)[:, 1]
        err = (f[:, 1]-f0[1])-(q[:, 1]-q0[1])
        return dict(base_y_m=s["base_delta_m"][1], body_vy_mean_m_s=body_vy.mean(),
                    yaw_end_deg=np.rad2deg(heading[-1]), yaw_peak_deg=max(abs(np.rad2deg(heading))),
                    hand_y_m=dy, loaded_hand_y_m=d[loaded, 1].sum(),
                    body_hand_y_change_m=local_hand[-1, 1]-hand0[1], tangent_lag_rms_m=np.sqrt((err**2).mean()),
                    duty=s["whole"]["duty"], rmse_N=s["whole"]["rmse_N"], traverse_rmse_N=s["walking"]["rmse_N"],
                    status=s["status"], **s["planning"])


def main():
    rows = []
    for case in [7, 9]:
        source = Path("artifacts/wall_layer_tick")/str(case)
        for mode in ["off", "conditioned", "matched"]:
            baseline = source/mode
            s = json.loads((baseline/"summary.json").read_text())
            out = Path("artifacts/uncertainty/wall")/str(case)/mode
            live = World(s["scene"], POLICY)
            pool = Rollouts(live, 8)
            result = episode(live, pool, s["desired_N"], s["distance_m"], mode, s["cem_seed"],
                             s["limit_s"], folder=out, sliding=True)
            (out/"summary.json").write_text(json.dumps(result, indent=2))
            for name, folder, summary in [("original", baseline, s), ("sliding", out, result)]:
                rows.append(dict(case=case, mode=mode, variant=name, **measure(folder, summary)))
            print(case, mode, rows[-1], flush=True)
    with Path("docs/wall_sliding_trials.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
