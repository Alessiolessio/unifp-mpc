"""Measure the executed trials and replay the force winners as position-only controls."""
import csv
import json

import numpy as np

from box_mpc import DT, WARMUP_TICKS, world_ee_target
from kinematics import nominal_targets, world_force
from force_box import MODES, POLICY, ROOT
from run_world import command_schedule
from unifp_native import World


def norm(x):
    return np.linalg.norm(x, axis=-1)


def measure(folder):
    s = json.loads((folder / "summary.json").read_text())
    case, mode = folder.parent.name, folder.name
    with np.load(folder / "episode.npz") as e, np.load(folder / "plans.npz") as p:
        root = e["qpos"][49:-1, :7]
        cmd = e["policy_command"][50:]
        raw = p["requested_prefix_m"].reshape(-1, 3)
        nominal = nominal_targets(root, raw)
        target = world_ee_target(root, cmd[:, 3:6])
        applied = world_force(root, cmd)
        effective = target + applied/200
        virtual = (effective - nominal)*200
        axis = e["goal_xy"] - s["box_start_xy_m"]
        axis /= norm(axis)
        desired = np.zeros((len(cmd), 2))
        if mode != "off":
            desired = p["requested_force_xy_N"].reshape(-1, 2)
            assert np.max(norm(np.diff(desired, axis=0))) <= 80*DT+1e-5
            assert np.max(norm(desired)) <= 40.00001
            assert np.max(norm(virtual[:, :2] - (virtual[:, :2] @ axis)[:, None]*axis)) < .001
            assert np.max(norm(virtual)-norm(desired)) < .001
            signed = virtual[:, :2] @ axis
            assert np.all(signed * (desired @ axis) >= -.001)
            for fraction in (.25, .5, 1.):
                intermediate = nominal + fraction*virtual/200
                np.testing.assert_allclose(nominal_targets(root, intermediate), intermediate,
                                           rtol=0, atol=1e-6)
        assert not cmd[:, 9:].any() and (mode == "conditioned" or not applied.any())
        force = e["gripper_on_box_N"][50:]
        touch = e["contact_duty"][50:] > 0
        issued_push = applied[:, :2] @ axis
        load = force[:, :2] @ axis
        progress = e["qpos"][50:, -7:-5] - s["box_start_xy_m"]
        return dict(case=case, mode=mode, status=s["status"], final_mm=1000*s["final_error_m"],
                    progress_pct=s["progress_pct"], displacement_cm=100*norm(progress[-1]),
                    control_s=len(root)*DT, settle_s=s["settle_times_s"]["precision"],
                    coarse_s=s["settle_times_s"]["coarse"], lateral_end_mm=1000*s["lateral_drift_m"],
                    peak_lateral_mm=1000*np.max(abs(progress @ [-axis[1], axis[0]])),
                    contact_s=DT*e["contact_duty"][50:].sum(),
                    first_contact_s=(np.flatnonzero(touch)[0]+1)*DT if touch.any() else None,
                    mean_touch_force_N=norm(force[touch]).mean() if touch.any() else 0,
                    peak_tick_force_N=norm(force).max(), signed_push_impulse_Ns=DT*(force[:, :2]@axis).sum(),
                    mean_touch_signed_command_N=issued_push[touch].mean() if touch.any() else None,
                    mean_touch_signed_load_N=load[touch].mean() if touch.any() else None,
                    peak_issued_step_N=np.max(abs(np.diff(issued_push))),
                    requested_abs_N=norm(desired).mean(), effective_offset_abs_N=norm(virtual).mean(),
                    issued_abs_N=norm(applied).mean(),
                    limited_pct=100*np.mean(norm(virtual) < norm(desired)-.01),
                    nominal_projected_pct=100*np.mean(norm(raw-nominal)>.00001),
                    effective_tracking_mm=1000*norm(effective-e["ee_world"][50:]).mean(),
                    tilt_deg=s["max_abs_roll_pitch_deg"], nongripper_Ns=s["other_robot_box_impulse_Ns"],
                    eligible=p["valid_candidates"].sum(), updates=s["mpc_updates"],
                    **s["planning"], prefix_error=s["prefix_max_error"])


def replay(folder):
    s = json.loads((folder / "summary.json").read_text())
    world = World(s["scene"], str(POLICY))
    world.run(command_schedule("push")[:WARMUP_TICKS])
    warm = world.snapshot()
    with np.load(folder / "episode.npz") as e, np.load(folder / "plans.npz") as p:
        cmd = e["policy_command"][50:]
        roots = e["qpos"][49:-1, :7]
        base = cmd[:, :3]
        targets = world_ee_target(roots, cmd[:, 3:6]) + world_force(roots, cmd)/200
        desired = np.column_stack((p["requested_prefix_m"].reshape(-1, 3), base,
                                   p["requested_force_xy_N"].reshape(-1, 2)))
        action_delta = []
        residual = []
        for i, request in enumerate(desired):
            snap = world.snapshot()
            matched = world.run(request[None], matched_position=True)
            world.restore(snap)
            conditioned = world.run(request[None])
            np.testing.assert_array_equal(conditioned["qpos"][0], e["qpos"][50+i])
            np.testing.assert_array_equal(conditioned["policy_command"][0], cmd[i])
            mc = matched["policy_command"][0]
            np.testing.assert_array_equal(mc[:3], cmd[i, :3])
            assert not mc[6:].any()
            assert .15-1e-6 <= mc[3] <= .55+1e-6
            assert np.deg2rad(-40)-1e-6 <= mc[4] <= np.deg2rad(50)+1e-6
            assert abs(mc[5]) <= np.pi/2+1e-6
            residual.append(norm(world_ee_target(roots[i], mc[3:6])-targets[i]))
            if norm(cmd[i, 6:9]) > 1:
                action_delta.append(abs(matched["action"] - conditioned["action"]).max())
        assert max(residual) < 1e-6
        world.restore(warm)
        local_replay = world.run(desired, matched_position=True)
        local_error = norm(local_replay["qpos"][-1, -7:-5] - e["goal_xy"])
        world.restore(warm)
        # Keep the recorded world effective path and all base requests fixed. Later poses
        # can diverge, so independently measure fresh clipping in the replay.
        t = world.run(np.column_stack((targets, base)))
        pre = np.vstack((roots[0], t["qpos"][:-1, :7]))
        actual = world_ee_target(pre, t["policy_command"][:, 3:6])
        gap = norm(actual-targets)
        axis = e["goal_xy"] - s["box_start_xy_m"]
        axis /= norm(axis)
        return dict(case=folder.parent.name, conditioned_error_mm=1000*s["final_error_m"],
                    recomputed_match_error_mm=1000*local_error,
                    replay_error_mm=1000*norm(t["qpos"][-1, -7:-5] - e["goal_xy"]),
                    replay_push_impulse_Ns=DT*np.sum(t["gripper_on_box_N"][:, :2]@axis),
                    replay_clipped_pct=100*np.mean(gap > .00001), replay_clip_max_mm=1000*gap.max(),
                    identical_state_target_max_um=1e6*max(residual), force_ticks_gt_1N=len(action_delta),
                    median_peak_action_delta=np.median(action_delta) if action_delta else 0)


def write(name, rows):
    with (ROOT / name).open("w") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    folders = sorted(ROOT.glob("*kg"))
    write("trials.csv", [measure(case / mode) for case in folders for mode in MODES])
    write("replays.csv", [replay(case / "conditioned") for case in folders])
