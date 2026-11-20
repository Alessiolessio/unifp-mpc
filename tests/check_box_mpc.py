"""Matched box-push comparison and a small set of changes to goal, start and planner budget."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
from cem import DT, PREFIX, TIMES, interpolate


def trial(folder, name, *options):
    path = folder / name
    subprocess.run([sys.executable, "-B", "tools/box_mpc.py", "--out", str(path), *options],
                   check=True, stdout=subprocess.DEVNULL)
    result = json.loads((path / "summary.json").read_text())
    with np.load(path / "episode.npz") as trace:
        assert not np.any(trace["contact_duty"][:50])
        mobile = result["controller"] == "mobile"
        base = trace["policy_command"][:, :3]
        if not mobile:
            assert not base[:, 0].any()
        if not result.get("planar"):
            assert not base[:, 1].any()
        if result.get("base_control") not in ("yaw", "full", "stance"):
            assert not base[:, 2].any()
        assert not np.any(trace["policy_command"][:, 6:])
        speed_command = trace["policy_command"][50:, 0]
        error = np.linalg.norm(trace["qpos"][50:, -7:-5] - trace["goal_xy"], axis=1)
        if len(error):
            np.testing.assert_allclose(error[-1], result["final_error_m"])
        speed = np.linalg.norm(trace["qvel"][-PREFIX:, -6:-4], axis=1)
        held = (len(error) >= 25 and np.all(error[-25:] <= result["tolerance_m"])
                and speed.max() <= result["settle_speed_m_s"])
        assert result["success"] == held
        warmup = trace["qpos"][:50].copy()
    if result["controller"] in ("mpc", "mobile"):
        with np.load(path / "plans.npz") as plans:
            requested = plans["requested_prefix_m"]
            valid = plans["valid_candidates"] > 0
            nodes = np.linspace(0., result.get("horizon_s", .5), 3)
            expected = interpolate(plans["knots_m"][valid], TIMES[:PREFIX], nodes)
            np.testing.assert_allclose(requested, expected, atol=1e-7, rtol=0)
            steps = np.linalg.norm(np.diff(requested.reshape(-1, 3), axis=0), axis=1)
            assert np.max(steps) <= .5 * DT + 2e-7
            assert np.all(plans["prefix_max_error"][valid] == 0)
            if mobile:
                speeds = plans["requested_vx_m_s"]
                knots_vx = plans["knots_vx_m_s"][valid]
                np.testing.assert_array_equal(knots_vx[:, 1], knots_vx[:, 2])
                np.testing.assert_allclose(interpolate(knots_vx[..., None], TIMES[:PREFIX], nodes)[..., 0],
                                           speeds, atol=1e-7, rtol=0)
                np.testing.assert_array_equal(speeds.ravel(), speed_command)
                assert np.max(np.abs(speeds)) <= .35 + 1e-7
                assert np.max(np.abs(np.diff(speeds.ravel()))) <= .8 * DT + 1e-7
    print(f"{name:16s} {result['status']:12s} error {result['final_error_m']*100:5.2f} cm"
          f"  displacement {np.around(np.array(result['displacement_xy_m'])*100, 2)} cm"
          f"  {result['mpc_updates']:2d} replans  cycle p95 {result['cycle']['p95_ms']:.1f} ms",
          flush=True)
    return result, warmup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("artifacts/box_mpc"))
    out = parser.parse_args().out
    main_result, warmup = trial(out, "mpc")
    assert main_result["success"] and main_result["gripper_contact_ticks"] > 0
    assert main_result["other_robot_box_impulse_Ns"] == 0
    for name, options in (
        ("hold", ["--controller", "hold"]),
        ("scripted", ["--controller", "scripted"]),
        ("oblique", ["--goal", "1.13", ".02"]),
        ("larger_goal", ["--goal", "1.20", "0"]),
        ("second_seed", ["--seed", "1"]),
        ("two_iterations", ["--iterations", "2", "--candidates", "16"]),
    ):
        _, initial = trial(out, name, *options)
        np.testing.assert_array_equal(initial, warmup)

    # Only translate the free box in the named reset key. Keep robot, policy and physics fixed.
    further = out / "box_4cm_further.mjb"
    model = mujoco.MjModel.from_binary_path("artifacts/spot_box.mjb")
    q = model.joint("box_joint").qposadr.item()
    model.key("spawn").qpos[q] += .04
    mujoco.mj_saveModel(model, str(further))
    result, initial = trial(out, "further_start", "--scene", str(further), "--goal", "1.20", "0")
    np.testing.assert_array_equal(initial[:, :q], warmup[:, :q])
    np.testing.assert_allclose(result["box_start_xy_m"], [1.12, 0], atol=1e-8)


if __name__ == "__main__":
    main()
