"""Check episode isolation and exactly which trajectory the passive viewer receives."""

from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np

from box_mpc import BoxPlanner, warmup_target
from box_scenario import generate, scene_template
from cem import PREFIX, interpolate
from run_world import command_schedule
from unifp_native import Rollouts, World


def main():
    policy = "artifacts/encoder_actor.onnx"
    model = scene_template()
    with TemporaryDirectory() as tmp:
        folder = Path(tmp)
        scene = str(folder / "scene.mjb")
        live = World("artifacts/spot_box.mjb", policy)
        pool = Rollouts(live, 2)
        live.run(command_schedule("push")[:100])
        old = live.snapshot()
        raw = command_schedule("push")[:50]
        for tier, horizon in (("easy", .5), ("medium", 1.), ("hard", 1.5), ("oblique", .5)):
            s = generate(folder, 7, difficulty=tier, model=model)
            spawn = mujoco.MjModel.from_binary_path(scene)
            q = spawn.key("spawn").qpos
            for joint, body in (("base", "robot"), ("box_joint", "box")):
                i = spawn.joint(joint).qposadr.item()
                np.testing.assert_array_equal(q[i:i+2], s[f"{body}_xy_m"])
                angle = s[f"{body}_yaw_rad"]
                np.testing.assert_allclose(q[i+3:i+7],
                                           [np.cos(angle/2), 0., 0., np.sin(angle/2)])
            assert spawn.pair("box_ground").friction[0] == s["ground_mu"]
            assert spawn.geom("box_collision").friction[0] == s["hand_mu"]
            live = pool.new_episode(scene)
            try:
                pool.evaluate(old, raw[None])
            except ValueError as error:
                assert "different world" in str(error)
            else:
                raise AssertionError("A previous episode's snapshot was accepted")
            warmup = live.run(raw)
            fresh = World(scene, policy).run(raw)
            for k in fresh:
                np.testing.assert_array_equal(warmup[k], fresh[k])
            assert not warmup["contact_duty"].any() and not warmup["other_robot_box_impulse_Ns"].any()
            box = warmup["qpos"][-1, -7:-4]
            planner = BoxPlanner(live, box, warmup_target(warmup), 8, 2, 2, 17, mobile=True,
                                 objective="bounded", pool=pool, workspace_yaw=s["robot_yaw_rad"],
                                 horizon=horizon, base_control="stance" if tier == "oblique" else "forward")
            old = live.snapshot()
            prefix, _, plan = planner.plan(live, box[:2], box[:2] + s["goal_offset_m"],
                                           capture_path=True, root=warmup["qpos"][-1, :7])
            request, finger = plan["view_paths"]
            assert request.shape == finger.shape == (round(horizon/.02), 3)
            cmd = interpolate(plan["knots_m"], planner.times, planner.nodes)
            np.testing.assert_allclose(request, cmd, rtol=0, atol=2e-7)
            np.testing.assert_array_equal(request[:PREFIX], prefix[:, :3])
            controls = [request]
            axes = ["vx_m_s", "vy_m_s", "yaw_rad_s"] if tier == "oblique" else ["vx_m_s"]
            for axis in axes:
                controls.append(interpolate(plan["knots_" + axis][:, None],
                                            planner.times, planner.nodes))
            replay = pool.evaluate(old, np.column_stack(controls)[None])
            np.testing.assert_array_equal(finger, replay["ee_world"][0])
            np.testing.assert_array_equal(finger[:PREFIX], live.run(prefix)["ee_world"])
    print("Fresh and reused worlds agree. Old snapshots rejected. All winning viewer paths agree.")


if __name__ == "__main__":
    main()
