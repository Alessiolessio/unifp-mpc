"""Rescore the original seed 5/6 CEM batches and resample the final executed snapshot."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from box_mpc import BoxPlanner, WARMUP_TICKS, warmup_target
from cone_mpc import angle_deg, clearance_cost, cone_cost
from run_world import command_schedule
from unifp_native import Rollouts, World


def diagnose(folder):
    summary = json.loads((folder / "summary.json").read_text())
    assert summary["objective"] == "upright_release" and summary["status"] == "no_valid_plan"
    live = World(str(folder / "scene.mjb"), "artifacts/encoder_actor.onnx")
    pool = Rollouts(live, 8)
    warm = live.run(command_schedule("push")[:WARMUP_TICKS])
    q = warm["qpos"][-1]
    planner = BoxPlanner(live, q[-7:-4], warmup_target(warm), 16, 1, 8, summary["cem_seed"],
                         mobile=True, horizon=1., base_control="full", sumo_sampling=True,
                         workspace_yaw=summary["scenario"]["robot_yaw_rad"], pool=pool)
    rows = []

    def score(batch):
        original = cone_cost(batch)
        new = clearance_cost(batch)
        valid = np.isfinite(original)
        bad = batch["other_robot_cone_impulse_Ns"] > 0
        row = dict(t=round(tick * .1, 1), valid=int(valid.sum()),
                   native_valid=int(batch["valid"].sum()))
        if valid.any():
            i, j = np.argmin(original), np.argmin(new)
            gap = batch["chassis_cone_gap_m"].min(1)
            row.update(baseline=int(i), clearance=int(j),
                       old_min_mm=round(1000*gap[i], 1), new_min_mm=round(1000*gap[j], 1),
                       best_valid_gap_mm=round(1000*gap[valid].max(), 1),
                       old_final_tilt=round(angle_deg(batch["qpos"][i, -1, -4:]), 1),
                       new_final_tilt=round(angle_deg(batch["qpos"][j, -1, -4:]), 1))
        if bad.any():
            hits = (bad[bad.any(1)].argmax(1)+1)*.02
            row.update(first_collision_s=[round(hits.min(), 2), round(hits.max(), 2)])
        rows.append(row)
        return original

    with np.load(folder / "episode.npz") as saved:
        for tick in range(summary["mpc_updates"]):
            if tick == summary["mpc_updates"]-1:
                # Copy the pre-shift CEM distribution. Each alternate draws at the same live state.
                alternatives = []
                for seed in (13, 31, 47, 101):
                    alt = copy.copy(planner)
                    for attr in ("center", "previous", "mean", "sigma", "winner"):
                        setattr(alt, attr, getattr(planner, attr).copy())
                    alt.rng = np.random.default_rng(seed)
                    alternatives.append(alt)
            prefix, _, _ = planner.optimize(live, q[-7:-5], score)
            if prefix is None:
                assert tick == len(saved["requests"]) == summary["mpc_updates"]-1
                break
            np.testing.assert_array_equal(prefix, saved["requests"][tick])
            executed = live.run(prefix)
            q = executed["qpos"][-1]
            np.testing.assert_array_equal(q, saved["qpos"][54+5*tick])
        for alt in alternatives:
            alt.optimize(live, q[-7:-5], score)

    path = Path("artifacts/cone_clearance")
    path.mkdir(parents=True, exist_ok=True)
    (path / f"probe-{folder.name}.json").write_text(json.dumps(rows, indent=2) + "\n")
    nominal = rows[:-4]
    changed = [r for r in nominal if r.get("baseline") != r.get("clearance")]
    print(folder, len(changed), "changed winners,", "valid in four fresh batches:",
          [r["valid"] for r in rows[-4:]])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("episodes", type=Path, nargs="+")
    for episode in p.parse_args().episodes:
        diagnose(episode)
