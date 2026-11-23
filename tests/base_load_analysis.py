"""Replay force-conditioned winners and summarize the paired disturbance trials."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from base_load_mpc import DT, KD, PREFIX, World, command_schedule, effective_velocity, yaw_of
from base_load_trials import CASES


def replay(summary, trace):
    live = World(summary["scene"], "artifacts/encoder_actor.onnx")
    live.run(command_schedule("stand")[:50])
    start = live.snapshot()
    cmds = trace["policy_command"]
    # Match the actual float32 command sent to the policy, before the common load and deadband.
    matched = cmds.copy()
    matched[:, :2] += cmds[:, 9:11] / KD
    matched[:, 9:11] = 0
    actions, target_errors = [], []
    loads, poses = trace["load_world_N"], trace["qpos"]
    for i, load in enumerate(loads):
        live.set_torso_load(*load)
        snap = live.snapshot()
        yaw = yaw_of(live.base_pose[3:7])
        c, s = np.cos(yaw), np.sin(yaw)
        external = load @ np.array([[c, -s], [s, c]])
        b, m = (effective_velocity(v, external) for v in (cmds[i], matched[i]))
        target_errors.append(abs(b-m).max())
        alternative = live.run(matched[i:i+1])["action"][0]
        live.restore(snap)
        original = live.run(cmds[i:i+1])
        np.testing.assert_array_equal(original["qpos"][0], poses[i])
        if np.linalg.norm(cmds[i, 9:11]) > 1:
            actions.append(abs(alternative-original["action"][0]).max())
    live.restore(start)
    for i, load in enumerate(loads):
        live.set_torso_load(*load)
        live.run(matched[i:i+1])
    return dict(target_delta_max_m_s=float(max(target_errors)),
                paired_action_median=float(np.median(actions)) if actions else None,
                replay_final_error_m=float(np.linalg.norm(live.base_pose[:2]-summary["goal_xy_m"])))


def pulse_metrics(loaded, free):
    # For each pulse, compare with the same controller/seed running without the load.
    on = np.flatnonzero(np.any(loaded["load_world_N"] != 0, axis=1))
    np.testing.assert_array_equal(loaded["qpos"][:on[0]], free["qpos"][:on[0]])
    displacement = np.linalg.norm(loaded["qpos"][:, :2]-free["qpos"][:, :2], axis=1)
    speed_error = np.linalg.norm(loaded["prefix_vel"]-free["prefix_vel"], axis=1)
    time = loaded["time_s"][PREFIX-1::PREFIX]
    release = (on[-1]+1)*DT
    # Three entire post-release prefixes within 2 cm and 0.1 m/s of the unloaded twin.
    inside = ((displacement.reshape(-1, PREFIX).max(1) <= .02) & (speed_error <= .10)
              & (time-PREFIX*DT >= release-1e-9))
    held = np.flatnonzero(np.convolve(inside, np.ones(3), "valid") == 3)
    return dict(pulse_twin_max_m=float(displacement.max()),
                pulse_twin_max_loaded_m=float(displacement[on].max()),
                recovered_after_onset_s=float(time[held[0]+2]-on[0]*DT) if len(held) else None,
                recovered_after_release_s=float(time[held[0]+2]-release) if len(held) else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()
    rows, latencies = [], []
    for case in CASES:
        for path in sorted((args.root/case).glob("*/summary.json")):
            row = json.loads(path.read_text())
            mode = path.parent.name
            with np.load(path.parent/"episode.npz") as tr:
                if row["controller"] == "conditioned":
                    row.update(replay(row, tr))
                if row["pulse"]:
                    with np.load(args.root/"held_free"/mode/"episode.npz") as free:
                        row.update(pulse_metrics(tr, free))
                arrival = np.linalg.norm(row["delta_m"]) / row["pace_m_s"]
                i = np.abs(tr["time_s"] - arrival).argmin()
                row["scheduled_arrival_error_m"] = float(np.linalg.norm(
                    tr["qpos"][i, :2] - row["goal_xy_m"]))
                row["mean_virtual_shift_m_s"] = row["virtual_mean_N"] / KD
                latencies.extend(tr["latency_s"] * 1000)
            rows.append(dict(case=case, **row))
    with args.csv.open("w") as stream:
        fields = dict.fromkeys(k for row in rows for k in row)
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)
    print(len(rows), "episodes;", len(latencies), "plans. Pooled p50/p95/max ms",
          np.percentile(latencies, [50, 95, 100]))


if __name__ == "__main__":
    main()
