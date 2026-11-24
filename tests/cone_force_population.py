"""Re-score the same actual base/arm population with or without its sampled XYZ force."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from box_mpc import BoxPlanner
from cone_mpc import angle_deg, clearance_cost, run_cone
from unifp_native import Rollouts, World

ROOT = Path("artifacts/cone_force")


def compare(cem):
    folder = ROOT/"general-82"/f"{cem}-16-conditioned-3"
    s = json.loads((folder/"summary.json").read_text())
    args = SimpleNamespace(**s, horizon=1., seed=cem, seconds=12., out=None)
    live = World(str(folder/"scene.mjb"), "artifacts/encoder_actor.onnx")
    pool = Rollouts(live, 8)
    original = BoxPlanner._evaluate
    # First plan and then each simulated second. Read exactly the samples given to CEM.
    rows = []
    updates = 0
    with np.load(folder/"episode.npz") as e:
        angles = angle_deg(e["qpos"][49::5, -4:])

    def record(planner, state, knots):
        nonlocal updates
        u, b, timing = original(planner, state, knots)
        n = updates
        updates += 1
        if n % 10 == 0:
            z = u.copy()
            z[...,6:8] = 0
            for mode, batch in (("B3", b), ("C3", pool.evaluate(state,u,matched_position=True)),
                                ("Z", pool.evaluate(state,z)), ("A", pool.evaluate(state,u[...,:6]))):
                cost = clearance_cost(batch)
                ok = np.isfinite(cost)
                gains = angles[n]-angle_deg(batch["qpos"][...,-4:]).min(axis=1)
                selected = np.argmin(cost)
                rows.append(dict(cem=cem, update=n, tilt_start_deg=angles[n], mode=mode,
                                 eligible=ok.sum(), gripper_candidates=batch["gripper_touch"][ok].any(1).sum(),
                                 improve_10deg=(ok & (gains>10)).sum(),
                                 best_gain_deg=gains[ok].max() if ok.any() else None,
                                 winner=int(selected) if ok.any() else None,
                                 winner_gain_deg=gains[selected] if ok.any() else None))
        return u,b,timing

    with patch.object(BoxPlanner,"_evaluate",record):
        repeat = run_cone(args, folder, s["scenario"], live, pool)
    assert (repeat["final_angle_deg"],repeat["status"]) == (s["final_angle_deg"],s["status"])
    return rows


if __name__ == "__main__":
    rows = compare(43) + compare(67)
    with (ROOT/"populations.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=rows[0], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
