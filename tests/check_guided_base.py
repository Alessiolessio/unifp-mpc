"""Replay winning guided plans with identical EE/vx and substitute the pure stance vy/yaw."""

import argparse
import json
from pathlib import Path

import numpy as np

from box_mpc import WARMUP_TICKS, box_cost, stance_knots
from cem import BASE_SLEW, DT, PREFIX, interpolate
from unifp_native import Rollouts, World


def audit(folder):
    settings = json.loads((folder / "summary.json").read_text())
    assert settings["base_control"] == "guided" and settings["objective"] == "bounded"
    live = World(settings["scene"], "artifacts/encoder_actor.onnx")
    pool = Rollouts(live, 2)
    start = np.array(settings["box_start_xy_m"])
    path = np.array(settings["goal_xy_m"]) - start
    length = np.linalg.norm(path)
    axis = path / length
    nodes = np.linspace(0., settings["horizon_s"], 3)
    times = np.arange(round(nodes[-1] / DT)) * DT
    cap = np.array([.22, .50])
    deviations, differences = [], []
    with np.load(folder / "episode.npz") as episode, np.load(folder / "plans.npz") as plans:
        live.run(episode["policy_command"][:WARMUP_TICKS])
        poses, applied = episode["qpos"], episode["policy_command"]
        assert not applied[:, 6:].any()
        assert np.all(np.abs(applied[:, :3]) <= np.array([.35, *cap]) + 1e-7)
        assert np.max(np.abs(np.diff(applied[:, :3], axis=0))) <= BASE_SLEW * DT + 1e-7
        for i in np.flatnonzero(plans["valid_candidates"]):
            t = WARMUP_TICKS + i * PREFIX
            q = poses[t - 1]
            xy = q[-7:-5]
            subgoal = start + np.clip((xy - start) @ axis + .08, 0., length) * axis
            rates = stance_knots(q[:7], xy, subgoal, applied[t-1, 1:3] / cap, cap, nodes[1]) * cap
            channels = ("vx_m_s", "vy_m_s", "yaw_rad_s")
            base = np.column_stack([plans[f"knots_{k}"][i] for k in channels])
            np.testing.assert_allclose(base[0], applied[t-1, :3], atol=1e-7, rtol=0)
            np.testing.assert_array_equal(base[1], base[2])
            selected = interpolate(np.column_stack((plans["knots_m"][i], base)), times, nodes)
            fixed = selected.copy()
            fixed[:, 4:] = interpolate(rates, times, nodes)
            trace = pool.evaluate(live.snapshot(), np.stack((selected, fixed)).astype(np.float32))
            costs = box_cost(trace, xy, subgoal, mobile=True, bounded=True)
            error = np.sum((trace["qpos"][:, :, -7:-5] - subgoal)**2, axis=2)
            costs += .5 * (error.mean(1) - error[:, -1])
            costs[~np.isfinite(costs)] = np.inf
            np.testing.assert_allclose(costs[0], plans["cost"][i], rtol=1e-7, atol=1e-10)
            result = live.run(selected[:PREFIX].astype(np.float32))
            np.testing.assert_array_equal(result["qpos"], poses[t:t+PREFIX])
            np.testing.assert_array_equal(result["policy_command"], applied[t:t+PREFIX])
            deviations.append(base[1, 1:] - rates[1])
            differences.append(costs[1] - costs[0])
    delta = np.abs(np.asarray(deviations).reshape(-1, 2))
    gain = np.array(differences)
    return dict(case=folder.name, updates=len(gain),
                mean_abs_rate_delta=delta.mean(0).tolist() if len(delta) else None,
                changed_axes=(delta > 1e-6).sum(0).tolist(),
                selected_better=int(np.sum(gain > 1e-10)), stance_better=int(np.sum(gain < -1e-10)),
                tie=int(np.sum(np.abs(gain) <= 1e-10)), invalid_stance=int(np.sum(~np.isfinite(gain))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", nargs="+", type=Path)
    for folder in parser.parse_args().folders:
        print(json.dumps(audit(folder)), flush=True)
