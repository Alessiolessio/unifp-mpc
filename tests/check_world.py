"""Compare closed-loop motion and signed box contacts to the paired Isaac recordings."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from run_world import command_schedule, spherical
from unifp_native import World


def free_motion(artifacts, outputs):
    scene = str(artifacts / "spot.mjb")
    policy = str(artifacts / "encoder_actor.onnx")
    center = mujoco.MjModel.from_binary_path(scene).body("body").ipos.copy()
    traces = outputs / "validation/free_space_x_reset_verified/traces"
    means = {}
    print("Free space, MuJoCo / Isaac. Settled means from 3.76 through 5.98 seconds.")
    for case in ("stand", "walk", "reach", "x_+20N_force", "x_+20N_position"):
        with np.load(traces / f"seed_0_{case}.npz") as ref:
            command = ref["command"]
            if case in ("stand", "walk", "reach"):
                np.testing.assert_allclose(command_schedule(case), command, atol=2e-7)
            out = World(scene, policy).run(command)
            np.testing.assert_allclose(out["action"][0], ref["action"][0], atol=2e-5)
            # Isaac samples before each action. State from native tick i aligns with Isaac i+1.
            tick = np.flatnonzero(ref["time_s"] >= 3.75)
            pose, vel = out["qpos"][tick - 1], out["qvel"][tick - 1]
            w, x, y, z = pose[:, 3:7].T
            roll = np.arctan2(2 * (w*x + y*z), 1 - 2*(x*x + y*y))
            pitch = np.arcsin(np.clip(2 * (w*y - z*x), -1, 1))
            yaw = np.arctan2(2 * (w*z + x*y), 1 - 2*(y*y + z*z))
            dx, dy, dz = (out["ee_world"][tick - 1] - pose[:, :3]).T
            ee = np.column_stack((np.cos(yaw)*dx + np.sin(yaw)*dy,
                                  -np.sin(yaw)*dx + np.cos(yaw)*dy, dz)) - [.35, 0, .12]
            # Inverse quaternion rotation to body frame, then translate from link origin to CoM.
            xyz = pose[:, 4:7]
            cross = np.cross(xyz, vel[:, :3])
            body_v = vel[:, :3] + 2 * (np.cross(xyz, cross) - w[:, None]*cross)
            body_v += np.cross(vel[:, 3:6], center)
            ref_v = ref["base_velocity_body_m_s"][tick]
            ref_ee = ref["ee_relative_yaw_m"][tick]
            target = ref["effective_target_m"][tick]
            errors = [np.linalg.norm(pos - target, axis=1).mean() for pos in (ee, ref_ee)]
            speed = [np.linalg.norm(v[:, :2], axis=1).mean() for v in (body_v, ref_v)]
            tilt = np.rad2deg([np.abs([roll, pitch]).max(), np.abs(ref["roll_pitch_rad"][tick]).max()])
            means[case] = np.stack((ee.mean(0), ref_ee.mean(0)))
            print(f"{case:16s} vx {body_v[:, 0].mean():.4f}/{ref_v[:, 0].mean():.4f} m/s,"
                  f" planar speed {speed[0]:.4f}/{speed[1]:.4f} m/s,"
                  f" EE error {errors[0]:.4f}/{errors[1]:.4f} m,"
                  f" max tilt {tilt[0]:.2f}/{tilt[1]:.2f} deg,"
                  f" mean height {pose[:, 2].mean():.3f}/{ref['root_position_local_m'][tick, 2].mean():.3f} m")
            assert np.isfinite(out["qpos"]).all() and np.min(out["qpos"][:, 2]) > .3
            if case == "stand":
                assert speed[0] < .05
            if case == "walk":
                assert abs(body_v[:, 0].mean() - .3) < .05
            if case == "reach":
                assert errors[0] < .10
    for case, axis in (("reach", 1), ("x_+20N_force", 0), ("x_+20N_position", 0)):
        change = (means[case] - means["stand"])[:, axis]
        print(f"{case} EE shift vs stand on requested axis: {change[0]:.4f}/{change[1]:.4f} m")
        # The pretrained force input produces only a few cm of free-space displacement.
        assert change[0] > 0 and abs(change[0] - change[1]) < .02


def push(artifacts, outputs):
    scene = str(artifacts / "spot_box.mjb")
    model = mujoco.MjModel.from_binary_path(scene)
    joint = model.joint("box_joint")
    q, v = joint.qposadr.item(), joint.dofadr.item()
    mass = model.body("box_body").mass.item()
    for case in ("hold", "nominal", "box_1kg_position_matched", "box_1kg_20N"):
        command = command_schedule("push")
        if case == "hold":
            command[:] = command[0]
        elif case in ("nominal", "box_1kg_20N"):
            command[50:, 3:6] = spherical([.38, 0, -.20])
            command[50:, 6] = 20 if case == "box_1kg_20N" else 0
        out = World(scene, str(artifacts / "encoder_actor.onnx")).run(command)
        # Box CoM coincides with its free-joint origin. Forces are interval means in world axes.
        velocity = out["qvel"][:, v:v + 3]
        force = out["gripper_on_box_N"] + out["ground_on_box_N"] + mass * model.opt.gravity
        reaction_error = mass * np.diff(velocity, axis=0, prepend=np.zeros((1, 3))) - .02 * force
        np.testing.assert_allclose(out["other_robot_box_impulse_Ns"], 0, atol=1e-9)
        np.testing.assert_allclose(reaction_error, 0, atol=1e-8)
        # Reference logs pre-action from t=0 to 9.98. Match its reported window [1,10).
        pos = out["qpos"][:, q:q + 3]
        delta = pos[-2] - pos[49]
        active = slice(49, -1)
        hits = np.flatnonzero(out["contact_duty"] > 0)
        first = (hits[0]+1)*.02 if len(hits) else np.nan
        peak = out["peak_forward_N"][active].max()
        mean = out["gripper_on_box_N"][active, 0].mean()
        duty = out["contact_duty"][active].mean()
        print(f"{case:25s} xy {delta[0]:+.5f} {delta[1]:+.5f} m, first {first:.2f} s,"
              f" peak/mean +x {peak:.2f}/{mean:.2f} N, duty {duty:.1%},"
              f" max momentum residual {np.abs(reaction_error).max():.2g} N s")
        assert not out["contact_duty"][:50].any()
        if case == "hold":
            assert not len(hits) and np.linalg.norm(delta[:2]) < 1e-5
        if case.startswith("box_1kg"):
            ref = json.loads((outputs / "playground" / case / "run.json").read_text())
            box = ref["box_measurements"]
            print(f"    Isaac xy {box['box_forward_displacement_m']:+.5f}"
                  f" {ref['box_result']['displacement_after_command_m'][1]:+.5f} m,"
                  f" first {box['first_contact_reported_s']:.2f} s,"
                  f" peak/mean +x {box['peak_forward_contact_N']:.2f}/"
                  f"{box['mean_forward_contact_N']:.2f} N,"
                  f" duty {box['contact_fraction_above_1N']:.1%}")
            assert 1 < first < 3 and peak > 1 and delta[0] > .03


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("../unifp-spot"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    free_motion(args.artifacts, args.source / "outputs")
    push(args.artifacts, args.source / "outputs")
