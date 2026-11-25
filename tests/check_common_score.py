"""Cross-check the external evaluator against archived native executions."""
import json
from pathlib import Path

import numpy as np
from physical_score import score


def evaluate(task, trace, summary, mass):
    q = trace["qpos"]
    target = np.array(summary.get("goal_xy_m"))
    touch = (trace["gripper_touch"][50:] if task=="cone" else trace["contact_duty"][50:] > 0)
    return score(task, q[50:], trace["qvel"][50:], touch, trace[f"ground_on_{task}_N"][50:,2],
                 trace[f"other_robot_{task}_impulse_Ns"][50:], mass, q[49], target)


if __name__ == "__main__":
    count = 0
    for p in sorted(Path("artifacts/uncertainty").glob("*/*/*/summary.json")):
        s = json.loads(p.read_text())
        task = p.parents[2].name
        if task not in ("box", "cone"):
            continue
        with np.load(p.parent/"episode.npz") as tr:
            r = evaluate(task, tr, s, s["execution_kg"])
        assert r["strict"] == (s["status"]=="success"), p
        if task=="cone":
            assert r["upright_s"] == s["sumo_first_s"]
        count += 1
    print("Archived outcomes reproduced:", count)
