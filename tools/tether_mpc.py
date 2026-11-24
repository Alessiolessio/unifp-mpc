"""Transport a free box with a tension-only torso tether and velocity-command MPC."""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import json
from pathlib import Path
from time import perf_counter

import mujoco
import numpy as np

from base_load_mpc import KD, RateCEM, reference
from forecasts import pack, winner
from box_mpc import latency, yaw_of
from cem import DT, PREFIX
from prepare_scene import scene_template
from run_world import command_schedule
from unifp_native import Rollouts, World
from view_box import Playback, WindowClosed, open_window

# Mass (kg), floor friction, initial spare length (m), torso yaw and goal bearing (deg).
CASES = {
    "straight": (1., .30, .04, 0., 0.),
    "diagonal": (1., .30, .08, 0., 30.),
    "heavy": (3., .35, .12, 15., 0.),
    "reverse_diagonal": (2., .27, .10, 18., -38.),
    "side": (3., .40, .14, -15., 65.),
}
RADIUS = .05


def prepare(model, scene, case):
    mass, friction, slack, yaw, bearing = CASES[case]
    yaw = np.deg2rad(yaw)
    body = model.body("box_body")
    sides2 = (2*model.geom("box_collision").size)**2
    body.inertia[:] = mass*((np.roll(sides2, 1)+np.roll(sides2, 2))/12)
    body.mass[:] = mass
    model.pair("box_ground").friction[:2] = friction
    data = mujoco.MjData(model)
    mujoco.mj_setConst(model, data)
    limit = model.tendon_range[model.tendon("tow_cable").id, 1]
    pose = model.key("spawn").qpos
    pose[3:7] = [np.cos(yaw/2), 0., 0., np.sin(yaw/2)]
    eye = model.site("box_eye").pos
    hitch = model.site("tow_hitch").pos
    dz = pose[2] + hitch[2] - pose[-5] - eye[2]
    span = np.sqrt((limit - slack)**2 - dz**2)
    pose[-7:-5] = hitch[0] * np.array([np.cos(yaw), np.sin(yaw)]) - eye[:2] - [span, 0.]
    mujoco.mj_resetDataKeyframe(model, data, model.key("spawn").id)
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.ten_length[model.tendon("tow_cable").id], limit-slack)
    box, ground = model.geom("box_collision").id, model.geom("ground").id
    if any(box in c.geom and ground not in c.geom for c in data.contact):
        raise ValueError("Robot and tethered box overlap at spawn")
    scene.parent.mkdir(parents=True, exist_ok=True)
    mujoco.mj_saveModel(model, str(scene))
    angle = np.deg2rad(bearing)
    return .8 * np.array([np.cos(angle), np.sin(angle)]), limit


def costs(forecast, goal, start, times, limit, guide):
    delta = goal - start
    distance = np.linalg.norm(delta)
    ref = reference(start, delta, times+DT, .23) if distance > 1e-8 else goal
    err = forecast["qpos"][..., -7:-5] - ref
    sq = np.sum(err**2, axis=-1)
    cost = .5 * (sq.mean(1) + sq[:, -1])
    if guide:
        # A box cannot move until the hitch takes up slack on a useful side.
        # Fade guidance near the goal, where cable angle is poorly defined.
        cable = forecast["eye_to_hitch_m"][:, -1]
        span = np.sqrt(np.maximum(0., limit**2 - cable[:, 2]**2))
        aim = span[:, None] * delta / max(distance, 1e-8)
        cost += .15 * min(distance/.15, 1.)**2 * np.sum((cable[:, :2]-aim)**2, axis=1)
    blocked = (forecast["other_robot_box_impulse_Ns"] > 0) | forecast["gripper_touch"]
    blocked |= forecast["cable_blocked"]
    cost[~forecast["valid"] | blocked.any(axis=1)] = np.inf
    return cost


def episode(live, pool, mode, delta, limit, seed, seconds, folder, scene, case, guide=True,
            display=None, horizon=1., goal_xy=None, broad=False):
    warm = live.run(command_schedule("stand")[:50])
    if (warm["tension_N"].max() > 0 or warm["other_robot_box_impulse_Ns"].any()
            or warm["gripper_touch"].any()):
        raise RuntimeError("Warmup must finish without towing or direct robot contact")
    start = warm["qpos"][-1, -7:-5].copy()
    box = start
    goal = start + delta if goal_xy is None else np.asarray(goal_xy)
    delta = goal-start
    if display:
        display.play(warm, goal, None, None)
    basis = np.eye(2)[:0 if mode == "off" else 2]
    planner = RateCEM(mode, basis, seed, horizon, broad)
    winners = []
    traces, valid, times, offsets = [], [], [], []
    hold = 0
    status = "timeout"
    rejection = {}
    for _ in range(round(seconds / (DT*PREFIX))):
        begin = perf_counter()
        snapshot = live.snapshot()
        commands = planner.sample()
        pre_root = live.base_pose
        forecast = pool.evaluate(snapshot, commands)
        cost = costs(forecast, goal, box, planner.times, limit, guide)
        valid.append(int(np.isfinite(cost).sum()))
        if not valid[-1]:
            times.append(perf_counter()-begin)
            rejection = dict(native_failure=int((~forecast["valid"]).sum()),
                             cable_crossing=int(forecast["cable_blocked"].any(1).sum()),
                             direct_contact=int(((forecast["other_robot_box_impulse_Ns"] > 0) |
                                                 forecast["gripper_touch"]).any(1).sum()))
            status = "no_valid_plan"
            break
        best = planner.update(cost)
        prefix = commands[best, :PREFIX]
        winners.append(winner(forecast, best, commands, pre_root))
        path = forecast["qpos"][best, :, -7:-4].copy() if display else None
        times.append(perf_counter()-begin)
        trace = live.run(prefix)
        for k in ("qpos", "qvel", "action", "tension_N", "eye_to_hitch_m", "cable_blocked"):
            np.testing.assert_array_equal(trace[k], forecast[k][best, :PREFIX])
        traces.append(trace)
        offsets.append(planner.rates[best, :PREFIX, 3:] @ basis / KD)
        box = trace["qpos"][-1, -7:-5]
        if display:
            display.play(trace, goal, path, times[-1])
        error = np.linalg.norm(trace["qpos"][:, -7:-5]-goal, axis=1)
        speed = np.linalg.norm(trace["qvel"][:, -6:-4], axis=1)
        for settled in (error <= RADIUS) & (speed <= .08):
            hold = hold+1 if settled else 0
        if hold >= 25:
            status = "settled"
            break
    result = dict(task="tether", case=case, controller=mode, cem_seed=seed, guidance=guide, broad_proposals=broad,
                  horizon_s=horizon, rejected_last_batch=rejection,
                  status=status, scene=str(scene), valid=sum(valid), candidates=24*len(valid),
                  box_start_xy_m=start.tolist(), goal_xy_m=goal.tolist(), tolerance_m=RADIUS,
                  simulated_s=len(traces)*PREFIX*DT, final_error_m=float(np.linalg.norm(box-goal)),
                  **latency(times))
    folder.mkdir(parents=True, exist_ok=True)
    if traces:
        fields = ("qpos", "qvel", "action", "policy_command", "ee_world", "tension_N", "eye_to_hitch_m",
                  "gripper_on_box_N", "other_robot_box_impulse_Ns", "cable_blocked")
        tr = {k: np.concatenate([r[k] for r in traces]) for k in fields}
        pose = tr["qpos"]
        tension = tr["tension_N"].ravel()
        taut = tension > .1
        xy = pose[:, -7:-5]-start
        direction = delta / max(np.linalg.norm(delta), 1e-8)
        tilt = np.arccos(np.clip(1-2*np.sum(pose[:, 4:6]**2, axis=1), -1., 1.))
        cable = tr["eye_to_hitch_m"]
        bearing = np.unwrap(np.arctan2(cable[:, 1], cable[:, 0]))
        loaded = (tr["tension_N"] > .1).any(axis=1)
        offset = np.concatenate(offsets)
        yaw = np.unwrap(yaw_of(pose[:, 3:7]))
        swing = np.ptp(bearing[loaded]) if loaded.any() else 0.
        virtual = np.linalg.norm(tr["policy_command"][:, 9:11], axis=1)
        result.update(progress_m=float(xy[-1] @ direction),
                      cross_track_max_m=float(abs(xy @ [-direction[1], direction[0]]).max()),
                      box_speed_max_m_s=float(np.linalg.norm(tr["qvel"][:, -6:-4], axis=1).max()),
                      tension_mean_N=float(tension.mean()), tension_peak_N=float(tension.max()),
                      taut_fraction=float(taut.mean()),
                      taut_entries=int(np.count_nonzero(taut & ~np.r_[False, taut[:-1]])),
                      cable_swing_deg=float(np.rad2deg(np.ptp(bearing))),
                      loaded_swing_deg=float(np.rad2deg(swing)),
                      max_stretch_mm=float(1000*max(0., np.linalg.norm(cable, axis=1).max()-limit)),
                      tilt_max_deg=float(np.rad2deg(tilt.max())),
                      yaw_range_deg=float(np.rad2deg(np.ptp(yaw))),
                      base_displacement_m=float(np.linalg.norm(pose[-1,:2]-warm["qpos"][-1,:2])),
                      hand_peak_N=float(np.linalg.norm(tr["gripper_on_box_N"], axis=1).max()),
                      body_box_impulse_Ns=float(tr["other_robot_box_impulse_Ns"].sum()),
                      virtual_mean_N=float(virtual.mean()),
                      effective_shift_mean_m_s=float(np.linalg.norm(offset, axis=1).mean()))
        np.savez_compressed(folder/"episode.npz", **tr, **pack(winners, 0), initial_qpos=warm["qpos"][-1],
                            equivalent_offset=offset, latency_s=times,
                            valid_count=valid, time_s=(np.arange(len(pose))+1)*DT)
    (folder/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", choices=(*CASES, "cycle"), default="cycle")
    p.add_argument("--mode", choices=("off", "conditioned", "matched"), default="off")
    p.add_argument("--seed", type=int, default=17, help="First CEM seed, incremented at reset")
    p.add_argument("--seconds", type=float, default=9.)
    p.add_argument("--goal", nargs=2, type=float, metavar=("X", "Y"),
                   help="Absolute world XY in metres")
    p.add_argument("--horizon", type=float, choices=(.5, 1.), default=1.)
    p.add_argument("--broad-proposals", action="store_true", help="Reserve six broad rate proposals")
    p.add_argument("--view", action="store_true", help="Continuous passive viewer")
    p.add_argument("--episodes", type=int, help="Stop after N episodes (one by default if headless)")
    p.add_argument("--out", type=Path, help="Keep every trace (viewer default keeps latest per case)")
    args = p.parse_args()
    if (not np.isfinite(args.seconds) or args.seconds < .1
            or not np.isclose(args.seconds*10, round(args.seconds*10))
            or args.episodes is not None and args.episodes < 1):
        p.error("Use positive episode counts and whole 0.1 s planning periods")
    if args.seed < 0 or (args.goal is not None and not np.isfinite(args.goal).all()):
        p.error("Expected a nonnegative seed and finite goal")
    root = args.out or Path("artifacts/live_tether" if args.view else "artifacts/tether")
    root.mkdir(parents=True, exist_ok=True)
    model = scene_template(tether=True)
    pool = display = None
    successes = i = 0
    with ExitStack() as stack:
        log = stack.enter_context((root/"episodes.jsonl").open("a", buffering=1))
        if args.view:
            visual = copy.copy(model)
            visual.geom("ground").group = 0
            data = mujoco.MjData(visual)
            win = stack.enter_context(open_window(visual, data))
            with win.lock():
                for flag in (mujoco.mjtRndFlag.mjRND_SHADOW, mujoco.mjtRndFlag.mjRND_REFLECTION):
                    win.user_scn.flags[flag] = 0
            executor = stack.enter_context(ThreadPoolExecutor(max_workers=1))
            display = Playback(win, data, executor, True, tether=True)
        try:
            while args.episodes is None or i < args.episodes:
                case = tuple(CASES)[i % len(CASES)] if args.case == "cycle" else args.case
                seed = args.seed+i
                keep = str(seed) if args.out or not args.view else "latest"
                folder = root/case/args.mode/keep
                scene = root/case/"scene.mjb"
                delta, limit = prepare(model, scene, case)
                if pool is None:
                    live = World(str(scene), "artifacts/encoder_actor.onnx")
                    pool = Rollouts(live, 8)
                else:
                    live = pool.new_episode(str(scene))
                if display:
                    label = f"Tether {case} | {args.mode} | seed {seed}"
                    display.begin(label, f"{successes}/{i} settled")
                r = episode(live, pool, args.mode, delta, limit, seed, args.seconds, folder, scene,
                            case, display=display, horizon=args.horizon, goal_xy=args.goal, broad=args.broad_proposals)
                log.write(json.dumps(r)+"\n")
                print(json.dumps(r), flush=True)
                successes += r["status"] == "settled"
                i += 1
                if display:
                    display.finish(r["status"], f"{successes}/{i} settled", .7)
                elif args.episodes is None:
                    break
        except WindowClosed:
            pass


if __name__ == "__main__":
    main()
