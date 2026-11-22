"""Check the imported cone, passive rest and contact-state branching on a saved episode."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np

from cone_mpc import CONE_XML, WARMUP_TICKS, angle_deg, cone_axis, cone_model, generate_cone
from run_world import command_schedule
from unifp_native import Rollouts, World

POLICY = "artifacts/encoder_actor.onnx"


def check_relative_spawns():
    model = cone_model()
    with TemporaryDirectory() as tmp:
        folder = Path(tmp)
        for tier, seed in (("varied", 72), ("general", 82)):
            s = generate_cone(folder, seed, model, tier)
            pose = model.key("spawn").qpos.copy()
            generate_cone(folder, seed, model, tier)
            np.testing.assert_array_equal(pose, model.key("spawn").qpos)
            np.testing.assert_allclose(s["spawn_floor_gap_m"], .005, atol=1e-12)
            plate = pose[-7:-5] - pose[:2]
            yaw = s["robot_yaw_rad"]
            np.testing.assert_allclose(plate, s["plate_range_m"] * np.array([
                np.cos(yaw + np.deg2rad(s["plate_bearing_deg"])),
                np.sin(yaw + np.deg2rad(s["plate_bearing_deg"]))]))
            axis = cone_axis(pose[-4:])
            az = yaw + np.deg2rad(s["relative_axis_bearing_deg"])
            np.testing.assert_allclose(axis[:2] / np.linalg.norm(axis[:2]), [np.cos(az), np.sin(az)])
            live = World(str(folder / "scene.mjb"), POLICY)
            warm = live.run(command_schedule("push")[:WARMUP_TICKS])
            assert not warm["gripper_touch"].any() and not warm["other_robot_cone_impulse_Ns"].any()
            np.testing.assert_allclose(angle_deg(warm["qpos"][-1, -4:]), 102.45, atol=.02)
            assert np.linalg.norm(warm["qvel"][-1, -6:]) < .001
            np.testing.assert_allclose(warm["ground_on_cone_N"][-1, 2], 3.36*9.81, atol=.01)


def check_rest():
    model = cone_model()
    source = mujoco.MjModel.from_xml_path(str(CONE_XML.resolve()))
    body = model.body("traffic_cone")
    for key in ("mass", "inertia", "ipos", "iquat"):
        np.testing.assert_array_equal(getattr(body, key), getattr(source.body("traffic_cone"), key))
    for i in np.flatnonzero(source.geom_bodyid == source.body("traffic_cone").id):
        original = source.geom(i)
        geom = model.geom(original.name)
        for key in ("type", "pos", "quat", "size", "friction", "priority", "solref", "solimp",
                    "condim", "contype", "conaffinity"):
            np.testing.assert_allclose(getattr(geom, key), getattr(original, key), atol=1e-12)
    joint = model.joint("traffic_cone_joint")
    assert joint.type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert joint.qposadr[0] == model.nq-7 and joint.dofadr[0] == model.nv-6
    np.testing.assert_array_equal(angle_deg(np.array([[1,0,0,0],[0,1,0,0]])), [0.,180.])
    with TemporaryDirectory() as tmp:
        folder = Path(tmp)
        generate_cone(folder, 0, model)
        spawn = model.key("spawn").qpos
        spawn[0] = -3.
        for upright in (False, True):
            if upright:
                spawn[-4:] = [1,0,0,0]
            path = folder / "rest.mjb"
            mujoco.mj_saveModel(model, str(path))
            live = World(str(path), POLICY)
            neutral = np.tile(command_schedule("push")[0], (150, 1))
            trace = live.run(neutral)
            assert not trace["contact_duty"].any() and not trace["other_robot_cone_impulse_Ns"].any()
            assert not trace["gripper_on_cone_N"].any()
            error = angle_deg(trace["qpos"][-25:, -4:])
            if upright:
                assert error.max() < .01
            else:
                assert np.all((100. < error) & (error < 105.))
            support = trace["ground_on_cone_N"][-25:, 2]
            np.testing.assert_allclose(support, float(body.mass[0]) * 9.81, atol=1e-4)
            assert np.max(np.abs(trace["qvel"][-25:, -6:])) < .0001
            print("Upright" if upright else "Fallen", "rest:", error[-1], "deg,", support[-1], "N")


def check_contact(folder):
    s = json.loads((folder / "summary.json").read_text())
    live = World(str(folder / "scene.mjb"), POLICY)
    model = mujoco.MjModel.from_binary_path(str(folder / "scene.mjb"))
    data = mujoco.MjData(model)
    with np.load(folder / "episode.npz") as e:
        requests = e["requests"].reshape(-1, 6)
        assert not e["policy_command"][:, 6:].any()
        np.testing.assert_array_equal(requests[:, 3:], e["policy_command"][WARMUP_TICKS:, :3])
        assert s["status"] == "success" and not e["gripper_touch"][-25:].any()
        assert np.all(e["ground_on_cone_N"][-25:, 2] >= .5*3.36*9.81)
        assert angle_deg(e["qpos"][-25:, -4:]).max() <= 10.
        assert np.linalg.norm(e["qvel"][-25:, -3:], axis=1).max() <= .25
        live.run(command_schedule("push")[:WARMUP_TICKS])
        tick = np.flatnonzero(e["contact_duty"][WARMUP_TICKS:])[0] + 1
        live.run(requests[:tick])
        state = live.snapshot()
        commands = requests[tick:tick+25]
        alt = commands.copy()
        alt[:, 3] = -alt[:, 3]
        batch = np.stack((commands, alt, commands))
        serial = Rollouts(live, 1).evaluate(state, batch)
        parallel = Rollouts(live, 3).evaluate(state, batch[[1, 2, 0]])
        assert serial["valid"].all() and parallel["valid"].all()
        executed = live.run(commands)
        live.restore(state)
        replay = live.run(commands)
        for key in executed:
            # The original archived demonstrations predate the clearance channel.
            if key != "chassis_cone_gap_m" or key in e:
                np.testing.assert_array_equal(executed[key], e[key][50+tick:50+tick+len(commands)])
            np.testing.assert_array_equal(executed[key], replay[key])
            np.testing.assert_array_equal(executed[key], serial[key][0])
            np.testing.assert_array_equal(serial[key][0], serial[key][2])
            np.testing.assert_array_equal(serial[key], parallel[key][[2, 0, 1]])
        chassis, cone = [], []
        for g in range(model.ngeom):
            if not (model.geom_contype[g] or model.geom_conaffinity[g]):
                continue
            body = model.body(model.geom_bodyid[g]).name
            if body == "body" or "_leg" in body:
                chassis.append(g)
            elif body == "traffic_cone":
                cone.append(g)
        for pose, gap in zip(executed["qpos"], executed["chassis_cone_gap_m"]):
            data.qpos[:] = pose
            mujoco.mj_kinematics(model, data)
            direct = min(mujoco.mj_geomDistance(model, data, a, b, .12, None)
                         for a in chassis for b in cone)
            np.testing.assert_allclose(gap, max(0., direct), atol=1e-8)
        q = e["qpos"][50+tick]
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        axis = data.body("traffic_cone").xmat.reshape(3, 3)[:, 2]
        np.testing.assert_allclose(cone_axis(q[-4:]), axis, atol=1e-14)
        np.testing.assert_array_equal(cone_axis(q[-4:]), cone_axis(-q[-4:]))
        # The cone's priority selects the sliding coefficient for both counterpart types.
        obj = model.body("traffic_cone").id
        seen = set()
        for pose in e["qpos"][e["gripper_touch"]]:
            data.qpos[:] = pose
            mujoco.mj_forward(model, data)
            for c in data.contact:
                a, b = c.geom
                if obj not in model.geom_bodyid[c.geom]:
                    continue
                geom, other = (a, b) if model.geom_bodyid[a] == obj else (b, a)
                seen.add("floor" if model.geom_bodyid[other] == 0 else "robot")
                np.testing.assert_allclose(c.friction[:2], model.geom_friction[geom, 0])
            if len(seen) == 2:
                break
        assert seen == {"floor", "robot"}
        for i, finger in enumerate(e["fingers"]):
            np.testing.assert_array_equal(finger[:5], e["ee_world"][50+5*i:55+5*i])
        np.testing.assert_array_equal(e["targets"][:, :5], e["requests"][:, :, :3])
    print("Restored contact, serial/parallel forecasts and executed viewer prefixes agree exactly.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("episode", type=Path)
    episode = p.parse_args().episode
    check_relative_spawns()
    check_rest()
    check_contact(episode)
