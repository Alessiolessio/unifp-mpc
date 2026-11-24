"""Replace only the four broad lanes at the same executed snapshots with their local draws."""
from copy import deepcopy
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import box_mpc
from cone_mpc import angle_deg, clearance_cost, run_cone
from unifp_native import Rollouts, World


def audit(scene, seed):
    folder = Path(f"artifacts/exploration/{scene}/{seed}-off-broad")
    summary = json.loads((folder/"summary.json").read_text())
    with np.load(folder/"episode.npz") as trace:
        tilt = angle_deg(trace["qpos"][49::5,-4:])
    live = World(str(folder/"scene.mjb"),"artifacts/encoder_actor.onnx")
    pool = Rollouts(live,8)
    args = SimpleNamespace(**(summary | dict(seed=seed, seconds=12., horizon=1., out=None)))
    original_sample, original_evaluate = box_mpc.sample, box_mpc.BoxPlanner._evaluate
    alternate = None
    rows = []
    update = 0

    def sample(*args, **options):
        nonlocal alternate
        if update % 10 == 0:
            local = list(args)
            local[6] = deepcopy(args[6])
            alternate = original_sample(*local, **(options | {"proposals": None}))
        return original_sample(*args, **options)

    def evaluate(planner, state, knots):
        nonlocal update
        commands, b, timing = original_evaluate(planner, state, knots)
        if update % 10 == 0:
            np.testing.assert_array_equal(alternate[:-4], knots[:-4])
            _, local, _ = original_evaluate(planner, state, alternate[-4:])
            local_cost = clearance_cost(local)
            broad_cost = clearance_cost(b)
            for mode, trace, cost, lanes in (("replacement",b,broad_cost,slice(-4,None)),
                                            ("original",local,local_cost,slice(None))):
                eligible = np.isfinite(cost[lanes])
                gain = tilt[update] - angle_deg(trace["qpos"][lanes,...,-4:]).min(1)
                rows.append(dict(scene=scene, cem=seed, update=update, lanes=mode,
                                 eligible=eligible.sum(),
                                 contact=(eligible & trace["gripper_touch"][lanes].any(1)).sum(),
                                 productive=(eligible & (gain >= 10)).sum(),
                                 best_gain_deg=gain[eligible].max() if eligible.any() else None,
                                 beats_other_population=bool(cost[lanes].min() < min(
                                     broad_cost[:-4].min(), (local_cost if mode=="replacement"
                                                              else broad_cost[-4:]).min()))))
        update += 1
        return commands,b,timing

    with patch.object(box_mpc,"sample",sample), patch.object(box_mpc.BoxPlanner,"_evaluate",evaluate):
        s = run_cone(args,folder,summary["scenario"],live,pool)
    assert s["final_angle_deg"] == summary["final_angle_deg"]
    return rows


if __name__ == "__main__":
    rows = audit(82,101) + audit(418,131)
    with Path("docs/exploration_batches.csv").open("w") as f:
        writer = csv.DictWriter(f,fieldnames=rows[0],lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
