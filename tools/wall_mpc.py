"""Whole-body MPC tracking physical gripper load on the fixed world-X wall."""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from functools import partial
from itertools import count
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter

import mujoco
import numpy as np

from forecasts import pack
from box_mpc import BoxPlanner, WARMUP_TICKS, latency, warmup_target, world_ee_target, yaw_of
from cem import DT, PREFIX
from cone_mpc import base_tilt
from prepare_scene import InvalidSpawn, scene_template
from run_world import command_schedule
from unifp_native import Rollouts, TrajectoryFailure, World

FACE = .89
POLICY = "artifacts/encoder_actor.onnx"


def wall_model(padded=False):
    model = scene_template(wall_face_x=FACE)
    if padded:
        face = model.geom("box_collision")
        face.solref[:] = [.06, 1.]
        face.solimp[:3] = [.005, .95, .05]
    return model


def spawn(model, path, seed):
    x, yaw = (0., 0.) if seed == 0 else np.random.default_rng(seed).uniform(
        [-.045, -.15], [.025, .15])
    q = model.key("spawn").qpos
    q[0] = x
    q[3:7] = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
    mujoco.mj_saveModel(model, str(path))
    return float(x), float(yaw)


def wall_cost(trace, desired, base_target, *, objective="tick", tangent_scale=.12):
    load = trace["gripper_on_wall_N"][..., 0]
    if objective == "tick":
        force = np.mean(((load - desired) / desired)**2, axis=1)
    else:
        windows = load.reshape(len(load), -1, PREFIX)
        force = np.mean(((windows.mean(axis=2) - desired) / desired)**2, axis=1)
        force += np.mean(np.maximum(1. - trace["wall_normal_N"]/desired, 0.), axis=(1, 2))
    # The finger body origin is behind its collision surface. Allow a 4 cm deadband.
    gap = np.maximum(FACE - .04 - trace["ee_world"][:, -1, 0], 0.)
    base = trace["qpos"][:, -1, :2] - base_target
    cost = force + (gap / .12)**2 + np.sum((base / [.12, tangent_scale])**2, axis=1)
    valid = trace["valid"] & (trace["other_robot_wall_impulse_Ns"].max(axis=1) == 0)
    return np.where(valid, cost, np.inf)


def force_metrics(trace, desired):
    samples = trace["wall_normal_N"]
    load = samples.mean(axis=1)
    error = load - desired
    return dict(mean_N=float(load.mean()), std_N=float(load.std()), mae_N=float(abs(error).mean()),
                rmse_N=float(np.sqrt(np.mean(error**2))), peak_N=float(samples.max()),
                physics_rmse_N=float(np.sqrt(np.mean((samples-desired)**2))),
                duty=float(np.mean(samples > 1.)), contact_s=float((samples > 1.).sum()*DT/4))


def walking_metrics(base, origin, distance, samples, force):
    # Inspect the actual traverse, excluding initial approach and final holding.
    progress = (base[:, 1] - origin[1]) / distance
    start = np.flatnonzero(progress >= .1)
    end = np.flatnonzero(progress >= .9)
    if not len(start):
        return None
    window = samples[start[0]:(end[0]+1 if len(end) else len(base))]
    return dict(reached_90pct=bool(len(end)), **force_metrics({"wall_normal_N": window}, force))


def episode(live, pool, force, distance, mode, seed, seconds, display=None, folder=None,
            horizon=.5, score=wall_cost):
    warm = live.run(command_schedule("push")[:WARMUP_TICKS])
    if warm["wall_normal_N"].any() or warm["other_robot_wall_impulse_Ns"].any():
        raise InvalidSpawn("Wall warmup must be contact-free")
    q0 = warm["qpos"][-1]
    goal = q0[:2] + [0., distance]
    anchor = warmup_target(warm)
    planner = BoxPlanner(live, np.array([FACE, q0[1], anchor[2]]), anchor, 24, 1, 8, seed,
                         pool=pool, record=bool(folder), mobile=True, base_control="full", moving_bounds=True, horizon=horizon,
                         workspace_offset=(-.1, 0., 0.), ee_force=mode, push_direction=[1., 0.])
    if display:
        display.play(warm, goal, None, None)
    start = live.snapshot() if folder and mode == "conditioned" else None
    winners = []
    elapsed, traces, schedules, paths, counts = [], [], [], [], []
    for _ in range(round(seconds / (PREFIX*DT))):
        before = perf_counter()
        base_y = live.base_pose[1]
        base_target = [goal[0], base_y + np.clip(goal[1]-base_y, -.12, .12)]
        prefix, forecast, plan = planner.optimize(
            live, np.array([FACE, base_y]), partial(score, desired=force, base_target=base_target),
            capture_path=bool(folder or display))
        elapsed.append(perf_counter()-before)
        counts.append(int(plan["valid_candidates"]))
        if prefix is None:
            break
        executed = live.run(prefix, matched_position=mode == "matched")
        if any(not np.array_equal(executed[k], v) for k, v in forecast.items()):
            raise RuntimeError("Live prefix diverged from wall forecast")
        traces.append(executed)
        schedules.append(prefix)
        if folder:
            winners.append(plan["forecast"])
            paths.append(plan["view_paths"])
        if display:
            display.play(executed, goal, plan["view_paths"], elapsed[-1])
    result = dict(task="wall", ee_force=mode, cem_seed=seed, desired_N=force, distance_m=distance,
                  goal_xy_m=goal.tolist(), planning=latency(elapsed), plans=len(elapsed),
                  eligible=sum(counts), horizon_s=horizon, limit_s=seconds,
                  control_s=len(traces)*PREFIX*DT,
                  status="no_valid_plan", prefix_exact=True)
    if not traces:
        return result
    t = {k: np.concatenate([row[k] for row in traces]) for k in warm}
    load = t["wall_normal_N"].mean(axis=1)
    np.testing.assert_allclose(load, t["gripper_on_wall_N"][:, 0], atol=1e-12)
    if (t["wall_normal_N"] < -1e-7).any():
        raise RuntimeError("Unexpected load opposite to the wall face normal")
    final = force_metrics({k: v[-50:] for k, v in t.items()}, force)
    base = t["qpos"][:, :2]
    tilt = base_tilt(t["qpos"])
    base_error = float(np.linalg.norm(base[-1]-goal))
    other = float(t["other_robot_wall_impulse_Ns"].sum())
    tolerance = max(1.5, .30*force)
    good = (abs(load-force) <= tolerance)
    hold = np.flatnonzero(np.convolve(good, np.ones(25), 'valid') == 25) if len(load) >= 25 else []
    first = np.flatnonzero(t["wall_normal_N"].ravel() > 1.)
    cmd = t["policy_command"]
    roots = np.vstack((q0, t["qpos"][:-1]))
    yaw = yaw_of(roots[:, 3:7])
    issued = np.cos(yaw)*cmd[:, 6] - np.sin(yaw)*cmd[:, 7]
    effective = world_ee_target(roots, cmd[:, 3:6])
    requests = np.concatenate(schedules)
    # The reward target includes the issued virtual-force offset in world coordinates.
    fy = np.sin(yaw)*cmd[:, 6] + np.cos(yaw)*cmd[:, 7]
    effective[:, :2] += np.column_stack((issued, fy)) / 200
    saturated = np.any(np.isclose(cmd[:, 3:6], [.15, np.deg2rad(-40), -np.pi/2], atol=1e-5) |
                       np.isclose(cmd[:, 3:6], [.55, np.deg2rad(50), np.pi/2], atol=1e-5), axis=1)
    full = len(traces) == round(seconds / (PREFIX*DT))
    walking = walking_metrics(base, q0, distance, t["wall_normal_N"], force) if distance else None
    walk_ok = not distance or (walking and walking["reached_90pct"] and
                               walking["duty"] >= .85 and walking["rmse_N"] <= tolerance)
    success = (full and walk_ok and len(load) >= 50 and final["rmse_N"] <= tolerance
               and final["duty"] >= .85 and base_error <= .075 and tilt.max() < 30. and other == 0)
    result.update(status="success" if success else "criteria_missed" if full else "no_valid_plan",
                  whole=force_metrics(t, force), last_second=final, walking=walking, base_error_m=base_error,
                  base_delta_m=(base[-1]-q0[:2]).tolist(),
                  max_cross_track_m=float(abs(base[:, 0]-q0[0]).max()),
                  max_tilt_deg=float(tilt.max()), other_impulse_Ns=other,
                  first_contact_s=float((first[0]+1)*DT/4) if len(first) else None,
                  first_half_second_in_band_s=float((hold[0]+25)*DT) if len(hold) else None,
                  issued_mean_N=float(issued.mean()), saturated_fraction=float(saturated.mean()),
                  tracking_rms_m=float(np.sqrt(np.mean(np.sum((effective-t["ee_world"])**2, axis=1)))),
                  finger_delta_m=(t["ee_world"][-1]-warm["ee_world"][-1]).tolist())
    if folder:
        folder.mkdir(exist_ok=True, parents=True)
        np.savez_compressed(folder/"episode.npz", **t, **pack(winners, 0),
                            warm_qpos=warm["qpos"], requests=requests,
                            paths=np.asarray(paths), planning_s=elapsed, initial_qpos=q0)
        # Counterfactual open loop: restore the whole warm state and replay the winning
        # requests with a reachable Cartesian offset instead of policy force conditioning.
        if mode == "conditioned":
            end = live.snapshot()
            live.restore(start)
            try:
                replay = live.run(requests, matched_position=True)
                result["matched_replay"] = force_metrics(replay, force)
                delta = replay["qpos"][-1, :2] - goal
                result["matched_replay"]["base_error_m"] = float(np.linalg.norm(delta))
            except TrajectoryFailure as exc:
                result["matched_replay"] = dict(failure=str(exc))
            finally:
                live.restore(end)
    return result


def main():
    from view_box import Playback, WindowClosed, open_window

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", type=float, default=5., help="Desired measured normal load (N)")
    p.add_argument("--distance", type=float, default=0., help="Signed travel along world Y (m)")
    p.add_argument("--ee-force", choices=("off", "conditioned", "matched"), default="off")
    p.add_argument("--scenario-seed", type=int, default=0)
    p.add_argument("--cem-seed", type=int, default=17)
    p.add_argument("--episodes", type=int, help="Default: repeat until closed")
    p.add_argument("--seconds", type=float)
    p.add_argument("--horizon", type=float, choices=(.5, 1.), default=.5)
    p.add_argument("--padded", action="store_true", help="Use a compliant fixed face")
    p.add_argument("--objective", choices=("tick", "impulse"), default="tick")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--log", type=Path, default=Path("artifacts/live_wall/episodes.jsonl"))
    args = p.parse_args()
    seconds = args.seconds if args.seconds is not None else (8. if args.distance else 5.)
    if (not np.isfinite(args.force) or args.force <= 0 or not np.isfinite(seconds) or seconds < .1
            or not np.isclose(seconds/.1, round(seconds/.1)) or abs(args.distance) > 1.
            or not np.isfinite(args.distance) or min(args.scenario_seed, args.cem_seed) < 0
            or args.episodes is not None and args.episodes < 1):
        p.error("Expected a positive setpoint, whole control prefixes and valid seeds/travel")
    model = wall_model(args.padded)
    pool = display = None
    args.log.parent.mkdir(exist_ok=True, parents=True)
    with ExitStack() as stack:
        path = Path(stack.enter_context(tempfile.TemporaryDirectory())) / "wall.mjb"
        log = stack.enter_context(args.log.open("a", buffering=1))
        if not args.headless:
            # The wall never moves. Only the robot spawn key changes between episodes.
            visual = copy.copy(model)
            visual.geom("ground").group = 0
            visual.geom("box_visual").rgba[3] = .40
            data = mujoco.MjData(visual)
            window = stack.enter_context(open_window(visual, data))
            executor = stack.enter_context(ThreadPoolExecutor(max_workers=1))
            display = Playback(window, data, executor, True, wall_force=args.force)
        wins = 0
        for i in count():
            seed = args.scenario_seed+i
            x, yaw = spawn(model, path, seed)
            if pool is None:
                live = World(str(path), POLICY)
                pool = Rollouts(live, min(8, len(os.sched_getaffinity(0))))
            else:
                live = pool.new_episode(str(path))
            try:
                if display:
                    display.begin(f"Wall   {args.ee_force}   scene {seed}   CEM {args.cem_seed}",
                                  f"{wins}/{i} successes")
                result = episode(live, pool, args.force, args.distance, args.ee_force, args.cem_seed,
                                 seconds, display, horizon=args.horizon,
                                 score=partial(wall_cost, objective=args.objective))
                wins += int(result["status"] == "success")
                if display:
                    display.finish(result["status"], f"{wins}/{i+1} successes", 1.)
            except WindowClosed:
                break
            except (InvalidSpawn, TrajectoryFailure) as exc:
                status = "invalid_spawn" if isinstance(exc, InvalidSpawn) else "live_failure"
                result = dict(task="wall", status=status, detail=str(exc), cem_seed=args.cem_seed,
                              ee_force=args.ee_force, desired_N=args.force, distance_m=args.distance)
            result.update(scene_seed=seed, spawn_x_m=x, yaw_rad=yaw, padded=args.padded,
                          objective=args.objective)
            print(json.dumps(result), file=log)
            timing = f"   planning p95 {result['planning']['p95_ms']:.1f} ms" if "planning" in result else ""
            print(f"Scene {seed}   {result['status']}   {wins}/{i+1} successes{timing}", flush=True)
            if args.episodes == i+1:
                break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
