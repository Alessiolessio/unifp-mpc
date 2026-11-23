"""Check unilateral tension against the airborne box momentum and fork across cable take-up."""
import argparse
import json
import tempfile
from pathlib import Path

import mujoco
import numpy as np

from base_load_mpc import KD, RateCEM, effective_velocity
from run_world import command_schedule
from tether_mpc import World, Rollouts, prepare, scene_template


def check():
    model = scene_template(tether=True)
    data = mujoco.MjData(model)
    with tempfile.TemporaryDirectory() as tmp:
        scene = Path(tmp)/"tether.mjb"
        prepare(model, scene, "straight")
        live = World(str(scene), "artifacts/encoder_actor.onnx")
        pool = Rollouts(live, 2)
        stand = command_schedule("stand")[:50]
        warm = live.run(stand)
        assert np.all(warm["tension_N"] == 0)
        pull = stand.copy()
        pull[:, 0] = .35
        snap = live.snapshot()
        forecast = pool.evaluate(snap, np.stack([pull, stand, pull]))
        actual = live.run(pull)
        assert actual["tension_N"].min() == 0 and actual["tension_N"].max() > 1
        for key in ("tension_N", "eye_to_hitch_m", "qpos", "qvel"):
            np.testing.assert_array_equal(actual[key], forecast[key][0])
            np.testing.assert_array_equal(actual[key], forecast[key][2])
        live.restore(snap)
        np.testing.assert_array_equal(live.run(stand)["qpos"], forecast["qpos"][1])
        # Lift both bodies off the floor and start just beyond the upper limit.
        pose = model.key("spawn").qpos
        pose[2] += 2.
        pose[-5] += 2.
        pose[-7] -= .05
        mujoco.mj_resetDataKeyframe(model, data, model.key("spawn").id)
        mujoco.mj_forward(model, data)
        assert data.ncon == 0
        d = data.site("tow_hitch").xpos - data.site("box_eye").xpos
        axis = d/np.linalg.norm(d)
        mujoco.mj_saveModel(model, str(scene))
        live = pool.new_episode(str(scene))
        tick = live.run(stand[:1])
        impulse = tick["tension_N"].sum() * model.opt.timestep
        mass = model.body("box_body").mass.item()
        measured = mass * (tick["qvel"][0, -6:-3] - .02*model.opt.gravity)
        np.testing.assert_allclose(measured, impulse*axis, atol=.002)
        print("Airborne box impulse (N s)", measured, "from tension", impulse*axis)
        print("Identical live/branched cable take-up; zero warmup tension.")


def inspect_trace(folder, disconnect):
    summary = json.loads((folder/"summary.json").read_text())
    model = mujoco.MjModel.from_binary_path(summary["scene"])
    data = mujoco.MjData(model)
    with np.load(folder/"episode.npz") as tr:
        poses, forces, commands = tr["qpos"], tr["tension_N"], tr["policy_command"]
    names = {}
    group = np.zeros(6, np.uint8)
    group[model.geom("box_collision").group.item()] = 1
    hit = np.zeros(1, np.int32)
    for pose in poses[forces.max(1) > .1]:
        data.qpos[:] = pose
        mujoco.mj_kinematics(model, data)
        eye = data.site("box_eye").xpos
        cable = data.site("tow_hitch").xpos - eye
        length = np.linalg.norm(cable)
        axis = cable/length
        distance = mujoco.mj_ray(model, data, eye+axis*1e-5, axis, group, 1, -1, hit)
        if 0 <= distance < length-2e-5:
            name = model.geom(hit[0]).name
            names[name] = names.get(name, 0)+1
    print(folder, "loaded policy ticks with a cable centreline intersection", names)
    if summary["controller"] == "conditioned":
        matched = commands.copy()
        matched[:, :2] += matched[:, 9:11]/KD
        matched[:, 9:11] = 0
        live = World(summary["scene"], "artifacts/encoder_actor.onnx")
        live.run(command_schedule("stand")[:50])
        origin = live.snapshot()
        i = int(np.argmax(forces.mean(1)))
        if i:
            live.run(commands[:i])
        same_state = live.snapshot()
        fork = live.run(commands[i:i+1])
        live.restore(same_state)
        partner = live.run(matched[i:i+1])
        # Same external term at a common state; target equivalence holds for any
        # shared value, including through the training deadband.
        np.testing.assert_allclose(effective_velocity(commands, [0., 0.]),
                                   effective_velocity(matched, [0., 0.]), atol=1e-7)
        live.restore(origin)
        replay = live.run(matched)
        print("Matched at peak tick, max action gap",
              abs(fork["action"]-partner["action"]).max(), "open-loop goal error (m)",
              np.linalg.norm(replay["qpos"][-1, -7:-5]-summary["goal_xy_m"]),
              "cable crossing in replay", replay["cable_blocked"].any())
    if summary["status"] == "no_valid_plan" and summary["controller"] == "off":
        live = World(summary["scene"], "artifacts/encoder_actor.onnx")
        pool = Rollouts(live, 8)
        live.run(command_schedule("stand")[:50])
        state = live.run(commands)
        np.testing.assert_array_equal(state["qpos"], poses)
        alternate = RateCEM("off", np.empty((0, 2)), 109, summary["horizon_s"])
        alternate.previous = commands[-1, :3] / alternate.scale
        alternate.sigma.fill(.8)
        batch = pool.evaluate(live.snapshot(), alternate.sample())
        legal = (batch["valid"] & ~batch["cable_blocked"].any(1) & ~batch["gripper_touch"].any(1)
                 & ~(batch["other_robot_box_impulse_Ns"] > 0).any(1))
        print("Restored stop, valid candidates in a fresh broad 24-sample batch", legal.sum())
        if legal.any():
            goal = summary["goal_xy_m"]
            print("Best legal predicted box error (m)",
                  np.linalg.norm(batch["qpos"][legal, -1, -7:-5]-goal, axis=1).min())
    if disconnect:
        model.tendon_limited[model.tendon("tow_cable").id] = 0
        with tempfile.TemporaryDirectory() as tmp:
            scene = str(Path(tmp)/"disconnected.mjb")
            mujoco.mj_saveModel(model, scene)
            live = World(scene, "artifacts/encoder_actor.onnx")
            start = live.run(command_schedule("stand")[:50])["qpos"][-1, -7:-5]
            replay = live.run(commands)
        distance = np.linalg.norm(replay["qpos"][-1, -7:-5]-start)
        print("Same commands, disconnected box displacement (m)", distance)
        assert not replay["tension_N"].any() and distance < 1e-5


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", nargs="*", type=Path)
    parser.add_argument("--disconnect", action="store_true")
    args = parser.parse_args()
    check()
    for folder in args.episodes:
        inspect_trace(folder, args.disconnect)
