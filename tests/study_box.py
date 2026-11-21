"""Run one planner configuration on paired scenarios and tabulate executed behavior."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

from box_scenario import generate
from cem import DT, PREFIX, interpolate


def measure(folder):
    s = json.loads((folder / "summary.json").read_text())
    with np.load(folder / "episode.npz") as e, np.load(folder / "plans.npz") as p:
        poses = e["qpos"]
        q = poses[50:]
        start = np.array(s["box_start_xy_m"])
        line = e["goal_xy"] - start
        axis = line / max(np.linalg.norm(line), 1e-12)
        travel = q[:, -7:-5] - start
        touched = e["contact_duty"][50:] > 0
        times = np.flatnonzero(touched)
        longest_gap = max(np.diff(times), default=1) - 1
        speeds = np.linalg.norm(e["qvel"][-5:, -6:-4], axis=1)
        errors = np.linalg.norm(q[:, -7:-5] - e["goal_xy"], axis=1)
        assert s["success"] == (len(q) >= 25 and max(errors[-25:]) <= .015 and max(speeds) <= .02)
        cmds = e["policy_command"][50:]
        assert not cmds[:, 6:].any()
        if s.get("base_control") in ("yaw", "full", "stance", "guided"):
            np.testing.assert_array_equal(cmds[:, 2], p["requested_yaw_rad_s"].ravel())
        else:
            assert not cmds[:, 2].any()
        if s.get("planar") or s.get("lateral_support"):
            np.testing.assert_array_equal(cmds[:, 1], p["requested_vy_m_s"].ravel())
        else:
            assert not cmds[:, 1].any()
        valid = p["valid_candidates"] > 0
        assert np.all(p["prefix_max_error"][valid] == 0)
        nodes = np.linspace(0., s.get("horizon_s", .5), 3)
        times_prefix = np.arange(PREFIX) * DT
        expected = interpolate(p["knots_m"][valid], times_prefix, nodes)
        np.testing.assert_allclose(expected, p["requested_prefix_m"], rtol=0., atol=2e-7)
        expected_vx = interpolate(p["knots_vx_m_s"][valid, :, None], times_prefix, nodes)
        np.testing.assert_allclose(expected_vx.ravel(), cmds[:, 0], rtol=0., atol=1e-7)
        requested = p["requested_prefix_m"].reshape(-1, 3)
        center = start + [-.22, 0.]
        near_bound = np.abs(requested[:, :2] - center) >= np.array([.42, .30]) - .01
        metrics = dict(scenario_seed=s["scenario"]["scenario_seed"], cem_seed=s["seed"],
                       status=s["status"], error_mm=1000*s["final_error_m"], control_s=len(q)*DT,
                       peak_tilt_deg=s["max_abs_roll_pitch_deg"], base_dx=s["base_displacement_xy_m"][0],
                       base_dy=s["base_displacement_xy_m"][1], contact_ticks=len(times),
                       first_contact_s=(times[0]+1)*DT if len(times) else np.nan,
                       intercontact_gap_s=longest_gap*DT,
                       last_gap_s=(len(q)-1-times[-1])*DT if len(times) else len(q)*DT,
                       nongripper_Ns=s["other_robot_box_impulse_Ns"],
                       cross_track_mm=max(abs(travel @ [-axis[1], axis[0]]), default=0.)*1000,
                       overshoot_mm=max(0., max(travel @ axis, default=0.) - np.linalg.norm(line))*1000,
                       radial_cap_fraction=np.mean(cmds[:, 3] >= .54999),
                       near_initial_bound_fraction=np.mean(near_bound.any(axis=1)),
                       updates=s["mpc_updates"], **s["planning"], cycle_p95_ms=s["cycle"]["p95_ms"],
                       cycle_misses=s["cycle"]["misses_100ms"])
        # Check the effective coefficients on real contact geometries from the executed episode.
        model = mujoco.MjModel.from_binary_path(s["scene"])
        data = mujoco.MjData(model)
        box, ground = model.geom("box_collision").id, model.geom("ground").id
        hand = {model.body(name).id for name in ("arm_link_fngr", "arm_link_wr1")}
        seen = set()
        for tick in [49, *(50 + times)]:
            data.qpos[:] = poses[tick]
            mujoco.mj_forward(model, data)
            for c in data.contact:
                if box not in c.geom:
                    continue
                other = c.geom[1] if c.geom[0] == box else c.geom[0]
                if other == ground or model.geom_bodyid[other] in hand:
                    key = "ground_mu" if other == ground else "hand_mu"
                    np.testing.assert_allclose(c.friction[:2], s["scenario"][key], atol=1e-10)
                    seen.add(key)
            if len(seen) == 2:
                break
        metrics["friction_pairs_checked"] = len(seen)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label")
    parser.add_argument("--cases", nargs="+", type=int, required=True)
    parser.add_argument("--cem-seeds", nargs="+", type=int, help="One optimizer seed per case")
    parser.add_argument("--root", type=Path, default=Path("artifacts/robust_box"))
    args, options = parser.parse_known_args()
    rows = []
    if args.cem_seeds and len(args.cem_seeds) != len(args.cases):
        parser.error("Supply one CEM seed per case")
    for index, seed in enumerate(args.cases):
        scene = args.root / "scenes" / str(seed)
        if not (scene / "scenario.json").exists():
            generate(scene, seed, long_push=seed in (3, 14))
        folder = args.root / args.label / str(seed)
        cmd = [sys.executable, "-B", "tools/box_mpc.py", "--controller", "mobile", "--out",
               str(folder), "--scenario", str(scene), *options]
        if args.cem_seeds:
            cmd.extend(["--seed", str(args.cem_seeds[index])])
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        row = dict(case=seed, **measure(folder))
        rows.append(row)
        print(f"{args.label} {seed}: {row['status']} {row['error_mm']:.1f} mm,"
              f" {row['control_s']:.1f} s, plan p95 {row['p95_ms']:.1f} ms", flush=True)
    with (args.root / args.label / "metrics.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
