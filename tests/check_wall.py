"""Check face geometry, forked virtual versus position commands and archived native loads."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from box_mpc import yaw_of, world_ee_target
from wall_mpc import FACE, POLICY, World, WARMUP_TICKS, command_schedule


def check(folder):
    result = json.loads((folder / "summary.json").read_text())
    scene = result["scene"]
    model = mujoco.MjModel.from_binary_path(scene)
    assert model.body("wall").jntnum.item() == 0
    face = model.geom("box_collision").id
    world = World(scene, POLICY)
    data = mujoco.MjData(model)
    warm = world.run(command_schedule("push")[:WARMUP_TICKS])
    maximum, differences, penetration, normals = 0., [], 0., []
    trace = np.load(folder / "episode.npz")
    np.testing.assert_array_equal(warm["qpos"], trace["warm_qpos"])
    for t, request in enumerate(trace["requests"]):
        root = world.base_pose
        snap = world.snapshot()
        matched = world.run(request[None], matched_position=True)
        world.restore(snap)
        actual = world.run(request[None])
        np.testing.assert_array_equal(actual["wall_normal_N"][0], trace["wall_normal_N"][t])
        cmd = actual["policy_command"][0]
        yaw = yaw_of(root[3:])
        fx, fy = cmd[6:8]
        reward = world_ee_target(root, cmd[3:6]) + np.array(
            [np.cos(yaw)*fx-np.sin(yaw)*fy, np.sin(yaw)*fx+np.cos(yaw)*fy, 0.])/200
        other = world_ee_target(root, matched["policy_command"][0, 3:6])
        maximum = max(maximum, np.linalg.norm(reward-other))
        if np.hypot(fx, fy) > 1.:
            differences.append(abs(actual["action"]-matched["action"]).max())
        data.qpos[:] = actual["qpos"][0]
        mujoco.mj_forward(model, data)
        for c in data.contact:
            if face in c.geom and model.geom("ground").id not in c.geom:
                # mjContact's first row points from geom[0] toward geom[1].
                normals.append(c.frame[0] if c.geom[1] == face else -c.frame[0])
                penetration = max(penetration, -c.dist)
                assert abs(c.pos[0]-FACE) < .05
    assert min(normals) > .999 and maximum < 1e-6
    action = np.median(differences) if differences else None
    print(folder, "minimum normal X", min(normals), "maximum contact depth", penetration,
          "matched target error (m)", maximum, "median max joint action difference", action)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("conditioned_episodes", type=Path, nargs="+")
    for folder in parser.parse_args().conditioned_episodes:
        check(folder)
