"""Closed-loop position-only UniFP box pushing in one live MuJoCo world."""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from cem import DT, NODES, PREFIX, TIMES, fit, interpolate, sample
from run_world import command_schedule
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


def box_cost(trace, current_xy, goal):
    pose = trace["qpos"][:, -1, -7:]
    cost = np.sum((pose[:, :2] - goal)**2, axis=1)
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
    return np.where(trace["valid"] & np.isfinite(cost), cost, np.inf)


class BoxPlanner:
    def __init__(self, live, box, start, count, iterations, workers, seed):
        self.pool = Rollouts(live, workers)
        self.center = box + [-.22, 0., .27]
        self.previous = (start - self.center) / SCALE
        if np.any(np.abs(self.previous) > 1):
            raise ValueError("Warmup EE request outside the box task workspace")
        self.mean = np.tile(self.previous, (len(NODES), 1))
        self.sigma = np.full_like(self.mean, .5)
        self.winner = self.mean.copy()
        self.rng = np.random.default_rng(seed)
        self.count = count
        self.iterations = iterations

    def plan(self, live, current_xy, goal):
        snapshot = live.snapshot()
        best_cost = np.inf
        mean, sigma, incumbent = self.mean, self.sigma, self.winner
        rollout_s = 0.
        valid_count = 0
        for _ in range(self.iterations):
            knots = sample(mean, sigma, incumbent, self.previous, SCALE, self.count, self.rng)
            commands = (self.center + SCALE * interpolate(knots, TIMES)).astype(np.float32)
            start = perf_counter()
            forecasts = self.pool.evaluate(snapshot, commands)
            rollout_s += perf_counter() - start
            costs = box_cost(forecasts, current_xy, goal)
            valid_count += np.isfinite(costs).sum()
            if not np.isfinite(costs).any():
                continue
            mean, sigma, index = fit(knots, costs)
            if costs[index] < best_cost:
                best_cost = costs[index]
                incumbent = knots[index].copy()
                prefix = commands[index, :PREFIX].copy()
                prediction = {k: forecasts[k][index, :PREFIX].copy()
                              for k in ("qpos", "action", "policy_command", "gripper_on_box_N")}
        if not np.isfinite(best_cost):
            return None, {}, dict(cost=np.inf, knots_m=np.full_like(self.mean, np.nan),
                                  valid_candidates=0, rollout_s=rollout_s)
        self.previous = (prefix[-1].astype(float) - self.center) / SCALE
        self.mean = interpolate(mean, NODES + PREFIX * DT)
        self.winner = interpolate(incumbent, NODES + PREFIX * DT)
        self.sigma = interpolate(sigma, NODES + PREFIX * DT)
        return prefix, prediction, dict(cost=best_cost, knots_m=self.center + SCALE * incumbent,
                                       valid_candidates=valid_count, rollout_s=rollout_s)


def latency(seconds):
    ms = np.asarray(seconds) * 1000
    return dict(p50_ms=float(np.median(ms)), p95_ms=float(np.percentile(ms, 95)),
                max_ms=float(ms.max()), misses_100ms=int(np.sum(ms > 100)))


def run(args):
    start = perf_counter()
    live = World(str(args.scene), str(args.policy))
    scripted = command_schedule("push")
    warmup = live.run(scripted[:WARMUP_TICKS])
    if warmup["qpos"].shape[1] != 33:
        raise ValueError("Expected the prepared floating Spot and free-box scene")
    box_start = warmup["qpos"][-1, -7:-4]
    goal = np.array(args.goal, dtype=float)
    anchor = warmup_target(warmup)
    planner = None
    if args.controller == "mpc":
        planner = BoxPlanner(live, box_start, anchor, args.candidates, args.iterations,
                             args.workers, args.seed)
    traces = [warmup]
    plans, cycles, requests = [], [], []
    in_goal = 0
    status = "time_limit"
    control_start = perf_counter()
    for update in range(int(args.seconds / (DT * PREFIX) + 1e-9)):
        cycle_start = perf_counter()
        if planner:
            commands, predicted, plan = planner.plan(live, traces[-1]["qpos"][-1, -7:-5], goal)
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
                   settle_speed_window_s=PREFIX*DT, seed=args.seed, scene=str(args.scene),
                   box_start_xy_m=box_start[:2].tolist(), displacement_xy_m=displacement.tolist(),
                   lateral_drift_m=float(displacement @ [-axis[1], axis[0]]),
                   final_error_m=float(np.linalg.norm(qpos[-1, -7:-5] - goal)), goal_hold_s=in_goal*DT,
                   success=status == "success", max_abs_roll_pitch_deg=float(tilt),
                   min_base_height_m=float(qpos[:, 2].min()),
                   final_box_speed_m_s=float(np.linalg.norm(trace["qvel"][-1, -6:-4])),
                   gripper_contact_ticks=int(np.count_nonzero(trace["contact_duty"][WARMUP_TICKS:])),
                   other_robot_box_impulse_Ns=float(trace["other_robot_box_impulse_Ns"].sum()),
                   mpc_updates=len(plans), simulated_s=len(qpos)*DT, wall_s=end-start,
                   setup_and_warmup_wall_s=control_start-start, control_wall_s=end-control_start,
                   wall_cycles_hz=(len(traces)-1)/(end-control_start), cycle=latency(cycles))
    if planner:
        summary.update(candidates=args.candidates, iterations=args.iterations, workers=args.workers,
                       planning=latency([p["planning_s"] for p in plans]),
                       rollout=latency([p["rollout_s"] for p in plans]),
                       prefix_max_error=max((float(p["prefix_max_error"]) for p in plans
                                             if p["valid_candidates"]), default=None))
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        np.savez_compressed(args.out / "episode.npz", **trace, goal_xy=goal,
                            time_s=(np.arange(len(qpos)) + 1)*DT, cycle_s=cycles)
        if plans:
            np.savez_compressed(args.out / "plans.npz", requested_prefix_m=requests,
                                **{k: [p[k] for p in plans] for k in plans[0]})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=("mpc", "hold", "scripted"), default="mpc")
    parser.add_argument("--goal", type=float, nargs=2, default=(1.16, 0.), metavar=("X", "Y"),
                        help="Absolute target for the box center in world metres")
    parser.add_argument("--seconds", type=float, default=9., help="Control duration after 1 s warmup")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=min(8, len(os.sched_getaffinity(0))))
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--scene", type=Path, default=Path("artifacts/spot_box.mjb"))
    parser.add_argument("--policy", type=Path, default=Path("artifacts/encoder_actor.onnx"))
    parser.add_argument("--out", type=Path, help="Directory for executed trace, summary and MPC plans")
    args = parser.parse_args()
    if (not np.isfinite(args.goal).all() or not np.isfinite(args.seconds) or args.seconds < .1
            or args.candidates < 3 or min(args.iterations, args.workers) < 1):
        parser.error("Expected finite goal, at least 0.1 s, three candidates and positive counts")
    run(args)
