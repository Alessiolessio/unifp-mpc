"""Replay one saved run and rescore identical native forecasts under all four objectives."""

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import box_mpc
from cone_mpc import angle_deg, clearance_cost, cone_cost, plate_access_cost, run_cone
from unifp_native import Rollouts, World


def audit(folder, sham):
    s = json.loads((folder / "summary.json").read_text())
    live = World(str(folder / "scene.mjb"), "artifacts/encoder_actor.onnx")
    pool = Rollouts(live, 8)
    args = SimpleNamespace(
        **{
            **s,
            "out": None,
            "seconds": s["control_limit_s"],
            "seed": s["cem_seed"],
            "horizon": s["horizon_s"],
        }
    )
    rows, ordinary = [], None
    sample, evaluate = box_mpc.sample, box_mpc.BoxPlanner._evaluate

    def paired_sample(
        mean,
        sigma,
        winner,
        previous,
        scale,
        count,
        rng,
        nodes,
        stance_rate=None,
        proposals=None,
    ):
        nonlocal ordinary
        if proposals is not None:
            ordinary = sample(
                mean,
                sigma,
                winner,
                previous,
                scale,
                count,
                copy.deepcopy(rng),
                nodes,
                stance_rate,
            )
        return sample(
            mean,
            sigma,
            winner,
            previous,
            scale,
            count,
            rng,
            nodes,
            stance_rate,
            None if sham else proposals,
        )

    def record(planner, state, knots):
        cmd, b, elapsed = evaluate(planner, state, knots)
        costs = [
            cone_cost(b, release=False),
            cone_cost(b),
            clearance_cost(b),
            plate_access_cost(b),
        ]
        winners = [int(np.argmin(c)) if np.isfinite(c).any() else -1 for c in costs]
        ok = np.isfinite(costs[0])
        angles = angle_deg(b["qpos"][..., -4:]).min(1)
        row = dict(
            winners=winners,
            eligible=int(ok.sum()),
            under90=int((ok & (angles < 90)).sum()),
            best=float(angles[ok].min()) if ok.any() else None,
            contact=bool(b["gripper_touch"][ok].any()),
        )
        if ordinary is not None and not sham:
            n = max(1, planner.count // 4)
            np.testing.assert_array_equal(ordinary[:-n], knots[:-n])
            # Only the four replaced lanes need a second rollout to recover the matched control.
            _, a, _ = evaluate(planner, state, ordinary[-n:])
            ac = plate_access_cost(a)
            aa = angle_deg(a["qpos"][..., -4:]).min(1)
            row.update(
                sub_winner=winners[-1] >= planner.count - n,
                beats_ordinary=bool(
                    np.min(costs[-1]) < min(np.min(costs[-1][:-n]), np.min(ac))
                ),
                paired_best=min(
                    [*angles[:-n][ok[:-n]], *aa[np.isfinite(ac)]], default=None
                ),
            )
        rows.append(row)
        return cmd, b, elapsed

    with (
        patch.object(box_mpc, "sample", paired_sample),
        patch.object(box_mpc.BoxPlanner, "_evaluate", record),
    ):
        result = run_cone(args, folder, s["scenario"], live, pool)
    if not sham:
        assert (
            result["status"] == s["status"]
            and result["final_angle_deg"] == s["final_angle_deg"]
        )
    out = folder / ("sham.json" if sham else "rankings.json")
    out.write_text(
        json.dumps(
            dict(
                rows=rows,
                status=result["status"],
                angle=result["final_angle_deg"],
                min_angle=result["min_angle_deg"],
                control_s=result["control_s"],
            )
        )
        + "\n"
    )
    print(out, result["status"], result["min_angle_deg"], result["control_s"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument(
        "--sham",
        action="store_true",
        help="Draw proposal randomness but issue the ordinary population",
    )
    a = p.parse_args()
    audit(a.folder, a.sham)
