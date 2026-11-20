"""Short matched locomotion/arm comparisons in the prepared 1 kg box scene."""

import argparse
import contextlib
import io
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

import box_mpc
from box_mpc import WARMUP_TICKS
from check_box_mpc import trial
from run_world import command_schedule
from unifp_native import World


def replay_channels(folder):
    with np.load(folder / "episode.npz") as trace, np.load(folder / "plans.npz") as plans:
        xyz = plans["requested_prefix_m"].reshape(-1, 3)
        vx = plans["requested_vx_m_s"].ravel()
        warmup = command_schedule("push")[:WARMUP_TICKS]
        base_only = np.repeat(warmup[-1:], len(vx), axis=0)
        base_only[:, 0] = vx
        live = World("artifacts/spot_box.mjb", "artifacts/encoder_actor.onnx")
        initial = live.run(warmup)["qpos"][-1]
        saved = live.snapshot()
        for name, commands in (("both", np.column_stack((xyz, vx))), ("arm_only", xyz),
                               ("base_only", base_only)):
            live.restore(saved)
            result = live.run(commands)
            if name == "both":
                for key in result:
                    np.testing.assert_array_equal(result[key], trace[key][WARMUP_TICKS:])
            error = np.linalg.norm(result["qpos"][-1, -7:-5] - trace["goal_xy"])
            assert (error <= .015) == (name == "both")
            delta = result["qpos"][-1] - initial
            print(f"{name:12s} replay  box XY {100*delta[-7:-5]} cm, base X {100*delta[0]:.1f} cm,"
                  f" goal error {100*error:.2f} cm, {np.count_nonzero(result['contact_duty'])}"
                  " gripper contact ticks")


def search_control(folder, *, mobile):
    # Keep the production loop, snapshot, RNG draws and evaluation budget in both controls.
    sampler = box_mpc.sample

    def sample(mean, sigma, winner, previous, scale, count, rng, nodes):
        if not mobile:
            mean = np.broadcast_to(previous, mean.shape)
            sigma = np.full_like(mean, .5)
            winner = mean
        knots = sampler(mean, sigma, winner, previous, scale, count, rng, nodes)
        if mobile:
            knots[..., 3] = 0.
        return knots

    name = "mobile_zero_vx" if mobile else "mpc_cold_start"
    args = argparse.Namespace(controller="mobile" if mobile else "mpc", warmup_ee=None,
                              goal=(1.43 if mobile else 1.16, 0.), iterations=1, candidates=32,
                              workers=8, seed=0, seconds=9., scene="artifacts/spot_box.mjb",
                              policy="artifacts/encoder_actor.onnx", out=folder / name)
    with patch.object(box_mpc, "sample", sample), contextlib.redirect_stdout(io.StringIO()):
        result = box_mpc.run(args)
    print(f"{name:16s} {result['status']}  error {100*result['final_error_m']:.2f} cm"
          f"  {result['mpc_updates']} replans")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("artifacts/mobile_box"))
    folder = parser.parse_args().out
    folder.mkdir(parents=True, exist_ok=True)
    scene = "artifacts/spot_box.mjb"
    model = mujoco.MjModel.from_binary_path(scene)
    q = model.joint("box_joint").qposadr.item()
    model.key("spawn").qpos[q:q+2] += [.02, .08]
    offset = folder / "offset.mjb"
    mujoco.mj_saveModel(model, str(offset))
    model = mujoco.MjModel.from_binary_path(scene)
    model.geom("box_collision").friction[0] = .5
    friction = folder / "higher_friction.mjb"
    mujoco.mj_saveModel(model, str(friction))

    for name, options in (("short", []), ("long", ["--goal", "1.43", "0"]),
                          ("lateral", ["--goal", "1.08", ".08"]),
                          ("offset_oblique", ["--scene", str(offset), "--goal", "1.35", ".04"])):
        baseline, initial = trial(folder, "mpc_" + name, *options)
        moving, matched = trial(folder, "mobile_" + name, "--controller", "mobile", *options)
        np.testing.assert_array_equal(initial, matched)
        assert moving["other_robot_box_impulse_Ns"] == 0
        if name == "long":
            assert moving["success"] and not baseline["success"]
            replay_channels(folder / "mobile_long")
    for name, options in (
        ("mobile_long_seed1", ["--controller", "mobile", "--seed", "1", "--goal", "1.43", "0"]),
        ("mobile_50cm", ["--controller", "mobile", "--goal", "1.58", "0"]),
        ("mpc_offset", ["--scene", str(offset), "--goal", "1.18", ".08"]),
        ("mpc_friction", ["--scene", str(friction)]),
        ("mpc_warmup_ee", ["--warmup-ee", ".20", ".09", ".10"]),
    ):
        trial(folder, name, *options)
    search_control(folder, mobile=True)
    search_control(folder, mobile=False)


if __name__ == "__main__":
    main()
