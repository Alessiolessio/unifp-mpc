"""Closed-loop UniFP box pushing with world EE targets and optional body-forward speed."""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from cem import DT, PREFIX, fit, interpolate, sample
from run_world import command_schedule, spherical
from unifp_native import Rollouts, World

RADIUS = .015
SETTLE_TICKS = 25
SETTLE_SPEED = .02
SCALE = np.array([.42, .30, .30])
WARMUP_TICKS = 50


def yaw_of(quat):
    w, x, y, z = np.moveaxis(quat, -1, 0)
    return np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))


def warmup_target(trace):
    root = trace["qpos"][-1, :7]
    r, pitch, az = trace["policy_command"][-1, 3:6]
    yaw = yaw_of(root[3:])
    return root[:3] + [.35*np.cos(yaw) + r*np.cos(pitch)*np.cos(az+yaw),
                       .35*np.sin(yaw) + r*np.cos(pitch)*np.sin(az+yaw), .12+r*np.sin(pitch)]


def box_cost(trace, current_xy, goal, mobile=False, bounded=False):
    pose = trace["qpos"][:, -1, -7:]
    cost = np.sum((pose[:, :2] - goal)**2, axis=1)
    terminal = cost.copy() if bounded else None
    remaining = np.linalg.norm(goal - current_xy)
    if remaining > RADIUS:
        direction = (goal - current_xy) / remaining
        yaw = yaw_of(pose[:, 3:])
        # Intersect the backwards goal ray with the yawed 0.40 m square footprint.
        c, s = np.cos(yaw), np.sin(yaw)
        dx, dy = direction
        radius = .20 / np.maximum(np.abs(c*dx + s*dy), np.abs(-s*dx + c*dy))
        face = pose[:, :3].copy()
        face[:, :2] -= radius[:, None] * direction
        face[:, 2] += .125
        cost += .05 * np.sum((trace["ee_world"][:, -1] - face)**2, axis=1)
    valid = trace["valid"]
    if mobile:
        root = trace["qpos"][:, -1, :7]
        yaw = yaw_of(root[:, 3:])
        offset = pose[:, :2] - root[:, :2]
        forward_gap = offset[:, 0] * np.cos(yaw) + offset[:, 1] * np.sin(yaw)
        # Leave forward working room for the arm as the base catches up with the box.
        cost += np.maximum(1.0 - forward_gap, 0.)**2
        valid = valid & (trace["other_robot_box_impulse_Ns"].max(axis=1) == 0)
    if bounded:
        # Approach breaks ties before contact. Bound its tradeoff as the box nears the goal.
        guide = cost - terminal
        budget = max(remaining**2, RADIUS**2)
        cost = terminal + budget * guide / (budget + guide)
    return np.where(valid & np.isfinite(cost), cost, np.inf)


class BoxPlanner:
    def __init__(self, live, box, start, count, iterations, workers, seed, mobile=False,
                 horizon=.5, objective="terminal", moving_bounds=False, planar=False, sumo_sampling=False):
        self.pool = Rollouts(live, workers)
        self.mobile = mobile
        self.nodes = np.linspace(0., horizon, 3)
        self.times = np.arange(round(horizon / DT)) * DT
        self.objective = objective
        self.moving_bounds = moving_bounds
        self.planar = planar
        self.sumo_sampling = sumo_sampling
        self.start_xy = box[:2].copy()
        self.center = box + [-.22, 0., .27]
        self.scale = SCALE
        if mobile:
            self.center = np.r_[self.center, 0.]
            self.scale = np.r_[SCALE, .35]
            start = np.r_[start, 0.]
        if planar:
            self.center = np.r_[self.center, 0.]
            self.scale = np.r_[self.scale, .22]
            start = np.r_[start, 0.]
        self.previous = (start - self.center) / self.scale
        if np.any(np.abs(self.previous) > 1):
            raise ValueError("Warmup EE request outside the box task workspace")
        self.mean = np.tile(self.previous, (len(self.nodes), 1))
        self.sigma = np.full_like(self.mean, .5)
        self.winner = self.mean.copy()
        self.rng = np.random.default_rng(seed)
        self.count = count
        self.iterations = iterations

    def plan(self, live, current_xy, goal):
        snapshot = live.snapshot()
        if self.moving_bounds:
            # Translate bounds while preserving the last issued request and warm knots in world metres.
            previous_xy = self.center[:2] + self.scale[:2] * self.previous[:2]
            center = np.clip(current_xy + [-.22, 0.], previous_xy - self.scale[:2],
                             previous_xy + self.scale[:2])
            shift = (self.center[:2] - center) / self.scale[:2]
            self.previous[:2] += shift
            self.mean[:, :2] += shift
            self.winner[:, :2] += shift
            self.center[:2] = center
        path = goal - self.start_xy
        length = np.linalg.norm(path)
        direction = path / max(length, 1e-12)
        final_target = True
        if self.mobile and length > 0.:
            progress = (current_xy - self.start_xy) @ direction
            # Limit the advance requested in one horizon and keep its target on the push line.
            final_target = progress + .08 >= length
            goal = self.start_xy + np.clip(progress + .08, 0., length) * direction
        best_cost = np.inf
        mean, sigma, incumbent = self.mean, self.sigma, self.winner
        rollout_s = 0.
        valid_count = 0
        for _ in range(self.iterations):
            if self.sumo_sampling:
                ramp = np.linspace(3.5 / len(self.nodes), 3.5, len(self.nodes))
                sigma = np.clip(sigma * ramp[:, None], .12, .8)
            knots = sample(mean, sigma, incumbent, self.previous, self.scale, self.count, self.rng,
                           self.nodes)
            normalized = interpolate(knots, self.times, self.nodes)
            commands = (self.center + self.scale * normalized).astype(np.float32)
            start = perf_counter()
            forecasts = self.pool.evaluate(snapshot, commands)
            rollout_s += perf_counter() - start
            bounded = self.objective in ("bounded", "settle")
            costs = box_cost(forecasts, current_xy, goal, self.mobile, bounded)
            xy = forecasts["qpos"][:, :, -7:-5]
            if self.objective == "path":
                cross_track = (xy - self.start_xy) @ [-direction[1], direction[0]]
                costs += 4 * np.mean(cross_track**2, axis=1)
            elif self.objective in ("running", "bounded", "speed", "settle"):
                error = np.sum((xy - goal)**2, axis=-1)
                costs += .5 * (np.mean(error, axis=1) - error[:, -1])
                if self.objective == "speed":
                    # Score box travel in one execution prefix at the predicted speed (m²).
                    vel = forecasts["qvel"][:, :, -6:-4]
                    costs += (PREFIX * DT)**2 * np.mean(np.sum(vel**2, axis=-1), axis=1)
                elif self.objective == "settle" and final_target:
                    vel = forecasts["qvel"][:, -1, -6:-4]
                    costs += (PREFIX * DT)**2 * np.sum(vel**2, axis=1)
            valid_count += np.isfinite(costs).sum()
            if not np.isfinite(costs).any():
                continue
            mean, sigma, index = fit(knots, costs, 3 if self.sumo_sampling else 2)
            if costs[index] < best_cost:
                best_cost = costs[index]
                incumbent = knots[index].copy()
                prefix = commands[index, :PREFIX].copy()
                prediction = {k: forecasts[k][index, :PREFIX].copy()
                              for k in ("qpos", "action", "policy_command", "gripper_on_box_N")}
        if np.isfinite(best_cost):
            self.previous = (prefix[-1].astype(float) - self.center) / self.scale
            shifted = self.nodes + PREFIX * DT
            self.mean = interpolate(mean, shifted, self.nodes)
            self.winner = interpolate(incumbent, shifted, self.nodes)
            self.sigma = interpolate(sigma, shifted, self.nodes)
            selected = self.center + self.scale * incumbent
        else:
            prefix, prediction = None, {}
            selected = np.full_like(self.mean, np.nan)
        plan = dict(cost=best_cost, knots_m=selected[:, :3], valid_candidates=valid_count,
                    rollout_s=rollout_s)
        if self.mobile:
            plan["knots_vx_m_s"] = selected[:, 3]
        if self.planar:
            plan["knots_vy_m_s"] = selected[:, 4]
        return prefix, prediction, plan


def latency(seconds):
    ms = np.asarray(seconds) * 1000
    return dict(p50_ms=float(np.median(ms)), p95_ms=float(np.percentile(ms, 95)),
                max_ms=float(ms.max()), misses_100ms=int(np.sum(ms > 100)))


def run(args, *, scenario=None, horizon=.5, objective="terminal", moving_bounds=False,
        planar=False, sumo_sampling=False):
    start = perf_counter()
    settings = json.loads((scenario / "scenario.json").read_text()) if scenario else None
    scene = scenario / "scene.mjb" if scenario else args.scene
    live = World(str(scene), str(args.policy))
    scripted = command_schedule("push")
    warmup_commands = scripted[:WARMUP_TICKS].copy()
    if args.warmup_ee is not None:
        request = spherical(args.warmup_ee)
        low = [.15, np.deg2rad(-40), -np.pi/2]
        high = [.55, np.deg2rad(50), np.pi/2]
        if np.any(request < np.array(low) - 1e-6) or np.any(request > np.array(high) + 1e-6):
            raise ValueError("Warmup EE request is outside UniFP's spherical command range")
        warmup_commands[:, 3:6] = request
    warmup = live.run(warmup_commands)
    if warmup["qpos"].shape[1] != 33:
        raise ValueError("Expected the prepared floating Spot and free-box scene")
    box_start = warmup["qpos"][-1, -7:-4]
    goal = box_start[:2] + settings["goal_offset_m"] if scenario else np.array(args.goal, dtype=float)
    if scenario and (warmup["contact_duty"].any() or warmup["other_robot_box_impulse_Ns"].any()):
        raise ValueError("Robot touched the box during scenario warmup")
    anchor = warmup_target(warmup)
    planner = None
    if args.controller in ("mpc", "mobile"):
        planner = BoxPlanner(live, box_start, anchor, args.candidates, args.iterations,
                             args.workers, args.seed, mobile=args.controller == "mobile",
                             horizon=horizon, objective=objective, moving_bounds=moving_bounds,
                             planar=planar, sumo_sampling=sumo_sampling)
    traces = [warmup]
    plans, cycles, requests = [], [], []
    in_goal = 0
    status = "time_limit"
    control_start = perf_counter()
    for update in range(int(args.seconds / (DT * PREFIX) + 1e-9)):
        cycle_start = perf_counter()
        if planner:
            q = traces[-1]["qpos"][-1]
            commands, predicted, plan = planner.plan(live, q[-7:-5], goal)
            plan["planning_s"] = perf_counter() - cycle_start
            if commands is None:
                status = "no_valid_plan"
                plan["prefix_max_error"] = np.nan
                plans.append(plan)
                cycles.append(perf_counter() - cycle_start)
                break
        elif args.controller == "hold":
            commands = np.tile(anchor, (PREFIX, 1))
        else:
            ticks = WARMUP_TICKS + update * PREFIX + np.arange(PREFIX)
            commands = scripted[np.minimum(ticks, len(scripted) - 1)]
        executed = live.run(commands)
        traces.append(executed)
        if planner:
            plan["prefix_max_error"] = max(np.max(np.abs(executed[k] - v))
                                           for k, v in predicted.items())
            if plan["prefix_max_error"] != 0:
                raise RuntimeError("Executed prefix diverged from its evaluated trajectory")
            plans.append(plan)
            requests.append(commands)
        errors = np.linalg.norm(executed["qpos"][:, -7:-5] - goal, axis=1)
        for error in errors:
            in_goal = in_goal + 1 if error <= RADIUS else 0
        speed = np.linalg.norm(executed["qvel"][:, -6:-4], axis=1)
        if in_goal >= SETTLE_TICKS and speed.max() <= SETTLE_SPEED:
            status = "success"
        cycles.append(perf_counter() - cycle_start)
        if status == "success":
            break

    end = perf_counter()
    trace = {k: np.concatenate([t[k] for t in traces]) for k in warmup}
    qpos = trace["qpos"]
    w, x, y, z = qpos[:, 3:7].T
    tilt = np.rad2deg(np.abs([np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
                              np.arcsin(np.clip(2*(w*y-z*x), -1, 1))])).max()
    displacement = qpos[-1, -7:-5] - box_start[:2]
    axis = goal - box_start[:2]
    axis /= max(np.linalg.norm(axis), 1e-12)
    summary = dict(controller=args.controller, status=status, goal_xy_m=goal.tolist(),
                   tolerance_m=RADIUS, required_hold_s=SETTLE_TICKS*DT, settle_speed_m_s=SETTLE_SPEED,
                   settle_speed_window_s=PREFIX*DT, seed=args.seed, scene=str(scene),
                   warmup_ee_yaw_frame_m=args.warmup_ee or [.20, 0., .10],
                   box_start_xy_m=box_start[:2].tolist(), displacement_xy_m=displacement.tolist(),
                   lateral_drift_m=float(displacement @ [-axis[1], axis[0]]),
                   final_error_m=float(np.linalg.norm(qpos[-1, -7:-5] - goal)), goal_hold_s=in_goal*DT,
                   success=status == "success", max_abs_roll_pitch_deg=float(tilt),
                   min_base_height_m=float(qpos[:, 2].min()),
                   base_displacement_xy_m=(qpos[-1, :2] - qpos[WARMUP_TICKS-1, :2]).tolist(),
                   final_box_speed_m_s=float(np.linalg.norm(trace["qvel"][-1, -6:-4])),
                   gripper_contact_ticks=int(np.count_nonzero(trace["contact_duty"][WARMUP_TICKS:])),
                   other_robot_box_impulse_Ns=float(trace["other_robot_box_impulse_Ns"].sum()),
                   mpc_updates=len(plans), simulated_s=len(qpos)*DT, wall_s=end-start,
                   setup_and_warmup_wall_s=control_start-start, control_wall_s=end-control_start,
                   wall_cycles_hz=(len(traces)-1)/(end-control_start), cycle=latency(cycles))
    if planner:
        summary.update(candidates=args.candidates, iterations=args.iterations, workers=args.workers,
                       horizon_s=horizon, objective=objective, moving_bounds=moving_bounds,
                       planar=planar, sumo_sampling=sumo_sampling,
                       planning=latency([p["planning_s"] for p in plans]),
                       rollout=latency([p["rollout_s"] for p in plans]),
                       prefix_max_error=max((float(p["prefix_max_error"]) for p in plans
                                             if p["valid_candidates"]), default=None))
    if settings is not None:
        summary["scenario"] = settings
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        np.savez_compressed(args.out / "episode.npz", **trace, goal_xy=goal,
                            time_s=(np.arange(len(qpos)) + 1)*DT, cycle_s=cycles)
        if plans:
            width = 3 + int(planner.mobile) + int(planner.planar)
            prefixes = np.asarray(requests).reshape(-1, PREFIX, width)
            fields = {k: [p[k] for p in plans] for k in plans[0]}
            if planner.mobile:
                fields["requested_vx_m_s"] = prefixes[..., 3]
            if planner.planar:
                fields["requested_vy_m_s"] = prefixes[..., 4]
            np.savez_compressed(args.out / "plans.npz", requested_prefix_m=prefixes[..., :3],
                                **fields)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=("mpc", "mobile", "hold", "scripted"), default="mpc")
    parser.add_argument("--warmup-ee", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        help="Initial Cartesian EE request relative to the yaw-frame sphere origin (m)")
    targets = parser.add_mutually_exclusive_group()
    targets.add_argument("--goal", type=float, nargs=2, default=(1.16, 0.), metavar=("X", "Y"),
                         help="Absolute target for the box center in world metres")
    targets.add_argument("--scenario", type=Path, help="Generated scene directory with a relative goal")
    parser.add_argument("--seconds", type=float, default=9., help="Control duration after 1 s warmup")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=min(8, len(os.sched_getaffinity(0))))
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--horizon", type=float, choices=(.5, 1., 1.5), default=.5)
    parser.add_argument("--objective", choices=("terminal", "running", "path", "bounded", "speed",
                                                "settle"), default="terminal")
    parser.add_argument("--sumo-sampling", action="store_true", help="Use three elites and a 3.5 noise ramp")
    parser.add_argument("--moving-bounds", action="store_true", help="Translate XY bounds with the live box")
    parser.add_argument("--planar", action="store_true", help="Sample body-lateral speed together with forward speed")
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--scene", type=Path, default=Path("artifacts/spot_box.mjb"))
    parser.add_argument("--policy", type=Path, default=Path("artifacts/encoder_actor.onnx"))
    parser.add_argument("--out", type=Path, help="Directory for executed trace, summary and MPC plans")
    args = parser.parse_args()
    if (not np.isfinite(args.goal).all() or not np.isfinite(args.seconds) or args.seconds < .1
            or (args.warmup_ee is not None and not np.isfinite(args.warmup_ee).all())
            or args.candidates < 3 or min(args.iterations, args.workers) < 1):
        parser.error("Expected finite goal, at least 0.1 s, three candidates and positive counts")
    if args.planar and args.controller != "mobile":
        parser.error("--planar requires --controller mobile")
    run(args, scenario=args.scenario, horizon=args.horizon, objective=args.objective,
        moving_bounds=args.moving_bounds, planar=args.planar, sumo_sampling=args.sumo_sampling)
