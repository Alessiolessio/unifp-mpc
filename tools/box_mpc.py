"""Closed-loop UniFP box pushing with world EE targets and body motion commands."""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from cem import BASE_SLEW, DT, PREFIX, fit, interpolate, refine_base, sample
from box_scenario import InvalidSpawn, episode_seconds
from run_world import command_schedule, spherical
from unifp_native import Rollouts, World

RADIUS = .015
SETTLE_TICKS = 25
SETTLE_SPEED = .02
SCALE = np.array([.42, .30, .30])
WARMUP_TICKS = 50
BASE_CONTROLS = ("forward", "lateral", "yaw", "full", "stance", "guided")
WORKING_DISTANCE = 1.0


def yaw_of(quat):
    w, x, y, z = np.moveaxis(quat, -1, 0)
    return np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))


def world_ee_target(root, spherical_command):
    """Decode the issued spherical command using the pre-action root pose."""
    r, pitch, az = np.moveaxis(spherical_command, -1, 0)
    yaw = yaw_of(root[..., 3:7])
    return root[..., :3] + np.stack((.35*np.cos(yaw) + r*np.cos(pitch)*np.cos(az+yaw),
                                     .35*np.sin(yaw) + r*np.cos(pitch)*np.sin(az+yaw),
                                     .12+r*np.sin(pitch)), axis=-1)


def warmup_target(trace):
    return world_ee_target(trace["qpos"][-1, :7], trace["policy_command"][-1, 3:6])


def stance_knots(root, box_xy, subgoal, previous, scale, knot_dt):
    """Normalized vy and yaw-rate knots, starting at the last executed body command."""
    yaw = yaw_of(root[3:7])
    left = np.array([-np.sin(yaw), np.cos(yaw)])
    offset = box_xy - root[:2]
    heading_error = np.arctan2(offset @ left, offset @ [np.cos(yaw), np.sin(yaw)])
    to_goal = subgoal - box_xy
    push = to_goal / max(np.linalg.norm(to_goal), RADIUS)
    # One metre behind the local push is the existing mobile task's working distance.
    side_error = (offset - WORKING_DISTANCE * push) @ left
    desired = np.clip(1.5 * np.array([side_error, heading_error]), -scale, scale)
    last = previous * scale
    rate = np.clip(desired, last - BASE_SLEW * knot_dt, last + BASE_SLEW * knot_dt)
    return np.stack((last, rate, rate)) / scale


def box_cost(trace, current_xy, goal, mobile=False, bounded=False, proximity=0.):
    pose = trace["qpos"][:, -1, -7:]
    cost = np.sum((pose[:, :2] - goal)**2, axis=1)
    terminal = cost.copy() if bounded else None
    remaining = np.linalg.norm(goal - current_xy)
    if remaining > RADIUS:
        finger = trace["ee_world"][:, -1]
        guide = np.zeros_like(cost)
        if proximity < 1:
            direction = (goal - current_xy) / remaining
            yaw = yaw_of(pose[:, 3:])
            # Intersect the backwards goal ray with the yawed 0.40 m square footprint.
            c, s = np.cos(yaw), np.sin(yaw)
            dx, dy = direction
            radius = .20 / np.maximum(np.abs(c*dx + s*dy), np.abs(-s*dx + c*dy))
            face = pose[:, :3].copy()
            face[:, :2] -= radius[:, None] * direction
            face[:, 2] += .125
            guide = (1 - proximity)*np.sum((finger - face)**2, axis=1)
        if proximity:
            guide += proximity*np.sum((finger - pose[:, :3])**2, axis=1)
        cost += .05 * guide
    valid = trace["valid"]
    if mobile:
        root = trace["qpos"][:, -1, :7]
        yaw = yaw_of(root[:, 3:])
        offset = pose[:, :2] - root[:, :2]
        forward_gap = offset[:, 0] * np.cos(yaw) + offset[:, 1] * np.sin(yaw)
        # Leave forward working room for the arm as the base catches up with the box.
        cost += np.maximum(WORKING_DISTANCE - forward_gap, 0.)**2
        valid = valid & (trace["other_robot_box_impulse_Ns"].max(axis=1) == 0)
    if bounded:
        # Approach breaks ties before contact. Bound its tradeoff as the box nears the goal.
        guide = cost - terminal
        budget = max(remaining**2, RADIUS**2)
        cost = terminal + budget * guide / (budget + guide)
    return np.where(valid & np.isfinite(cost), cost, np.inf)


class BoxPlanner:
    def __init__(self, live, box, start, count, iterations, workers, seed, mobile=False,
                 horizon=.5, objective="terminal", moving_bounds=False, planar=False,
                 sumo_sampling=False, pool=None, workspace_yaw=None, base_control=None,
                 ee_force="off", push_direction=None):
        self.pool = pool if pool is not None else Rollouts(live, workers)
        self.mobile = mobile
        self.nodes = np.linspace(0., horizon, 3)
        self.times = np.arange(round(horizon / DT)) * DT
        self.objective = objective
        self.moving_bounds = moving_bounds or workspace_yaw is not None
        self.base_control = base_control or ("lateral" if planar else "forward")
        if self.base_control not in BASE_CONTROLS or (self.base_control != "forward" and not mobile):
            raise ValueError("Select a supported base mode for the mobile controller")
        self.planar = self.base_control in ("lateral", "full", "stance", "guided")
        self.yaw_control = self.base_control in ("yaw", "full", "stance", "guided")
        self.sumo_sampling = sumo_sampling
        self.start_xy = box[:2].copy()
        self.rotation = None
        if workspace_yaw is not None:
            c, s = np.cos(workspace_yaw), np.sin(workspace_yaw)
            self.rotation = np.array([[c, -s], [s, c]])
            box, start = box.copy(), start.copy()
            box[:2] = box[:2] @ self.rotation
            start[:2] = start[:2] @ self.rotation
        self.center = box + [-.22, 0., .27]
        self.scale = SCALE
        if self.rotation is not None:
            self.center[:2] = np.clip(self.center[:2], start[:2] - SCALE[:2], start[:2] + SCALE[:2])
        if mobile:
            self.center = np.r_[self.center, 0.]
            self.scale = np.r_[SCALE, .35]
            start = np.r_[start, 0.]
        if self.planar:
            self.center = np.r_[self.center, 0.]
            self.scale = np.r_[self.scale, .22]
            start = np.r_[start, 0.]
        if self.yaw_control:
            self.center = np.r_[self.center, 0.]
            self.scale = np.r_[self.scale, .5]
            start = np.r_[start, 0.]
        self.ee_force = ee_force
        self.force_rng = None
        if ee_force != "off":
            if self.base_control != "full" or ee_force not in ("conditioned", "matched"):
                raise ValueError("EE force experiments require full mobile base control")
            if push_direction is None or np.shape(push_direction) != (2,):
                raise ValueError("Expected the world XY push direction for EE force")
            length = np.linalg.norm(push_direction)
            if not np.isfinite(length) or length == 0:
                raise ValueError("EE force experiments require a nonzero goal displacement")
            self.push_axis = np.asarray(push_direction) / length
            self.force_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
            self.center = np.r_[self.center, 0.]
            self.scale = np.r_[self.scale, 40.]
            start = np.r_[start, 0.]
        self.previous = (start - self.center) / self.scale
        if np.any(np.abs(self.previous) > 1 + 1e-12):
            raise ValueError("Warmup EE request outside the box task workspace")
        self.mean = np.tile(self.previous, (len(self.nodes), 1))
        self.sigma = np.full_like(self.mean, .5)
        self.winner = self.mean.copy()
        self.rng = np.random.default_rng(seed)
        self.count = count
        self.iterations = iterations

    def _evaluate(self, snapshot, knots):
        normalized = interpolate(knots, self.times, self.nodes)
        targets = self.center + self.scale * normalized
        if self.rotation is not None:
            targets[..., :2] = targets[..., :2] @ self.rotation.T
        commands = targets.astype(np.float32)
        if self.yaw_control and not self.planar:
            commands = np.insert(commands, 4, 0., axis=-1)
        if self.force_rng is not None:
            commands = np.concatenate((commands[..., :6],
                                       commands[..., 6:] * self.push_axis), axis=-1).astype(np.float32)
        start = perf_counter()
        if self.ee_force == "matched":
            forecasts = self.pool.evaluate(snapshot, commands, matched_position=True)
        else:
            forecasts = self.pool.evaluate(snapshot, commands)
        return commands, forecasts, perf_counter() - start

    def _box_cost(self, forecasts, current_xy, goal, direction, final_target, proximity):
        bounded = self.objective in ("bounded", "settle", "proximity", "hybrid")
        costs = box_cost(forecasts, current_xy, goal, self.mobile, bounded, proximity)
        xy = forecasts["qpos"][:, :, -7:-5]
        if self.objective == "path":
            cross_track = (xy - self.start_xy) @ [-direction[1], direction[0]]
            costs += 4 * np.mean(cross_track**2, axis=1)
        elif bounded or self.objective in ("running", "speed"):
            error = np.sum((xy - goal)**2, axis=-1)
            costs += .5 * (np.mean(error, axis=1) - error[:, -1])
            if self.objective == "speed":
                # Score box travel in one execution prefix at the predicted speed (m²).
                vel = forecasts["qvel"][:, :, -6:-4]
                costs += (PREFIX * DT)**2 * np.mean(np.sum(vel**2, axis=-1), axis=1)
            elif self.objective == "settle" and final_target:
                vel = forecasts["qvel"][:, -1, -6:-4]
                costs += (PREFIX * DT)**2 * np.sum(vel**2, axis=1)
        return costs

    def plan(self, live, current_xy, goal, *, finger_gap=None, capture_path=False, root=None):
        path = goal - self.start_xy
        length = np.linalg.norm(path)
        direction = path / max(length, 1e-12)
        final_target = True
        if self.mobile and length > 0.:
            progress = (current_xy - self.start_xy) @ direction
            # Limit the advance requested in one horizon and keep its target on the push line.
            final_target = progress + .08 >= length
            goal = self.start_xy + np.clip(progress + .08, 0., length) * direction
        proximity = float(self.objective == "proximity")
        if self.objective == "hybrid":
            proximity = np.clip((finger_gap - .35) / .20, 0., 1.)
        rates = None
        if self.base_control in ("stance", "guided"):
            if root is None:
                raise ValueError("Stance guidance needs the executed pre-action base pose")
            rates = stance_knots(root, current_xy, goal, self.previous[4:], self.scale[4:],
                                 self.nodes[1] - self.nodes[0])

        def score(forecasts):
            return self._box_cost(forecasts, current_xy, goal, direction, final_target, proximity)

        return self.optimize(live, current_xy, score, capture_path=capture_path, rates=rates)

    def optimize(self, live, current_xy, score, *, capture_path=False, rates=None, propose=None):
        """Score bounded UniFP commands from one live snapshot and shift the winning plan."""
        snapshot = live.snapshot()
        root = live.base_pose if capture_path else None
        if self.moving_bounds:
            # Translate bounds while preserving the last issued request and warm knots in world metres.
            previous_xy = self.center[:2] + self.scale[:2] * self.previous[:2]
            xy = current_xy if self.rotation is None else current_xy @ self.rotation
            center = np.clip(xy + [-.22, 0.], previous_xy - self.scale[:2],
                             previous_xy + self.scale[:2])
            shift = (self.center[:2] - center) / self.scale[:2]
            self.previous[:2] += shift
            self.mean[:, :2] += shift
            self.winner[:, :2] += shift
            self.center[:2] = center
        best_cost = np.inf
        mean, sigma, incumbent = self.mean, self.sigma, self.winner
        rollout_s = 0.
        valid_count = 0
        paths = None
        sampled = slice(None)
        stance = None
        stance_rate = None
        if rates is not None:
            if self.base_control == "stance":
                stance = np.broadcast_to(rates, (self.count, *rates.shape))
                sampled = slice(4)
            else:
                stance_rate = rates[1]
        batch_sizes = (self.count,)
        if stance_rate is not None and self.count >= 8:
            # Spend the last third on matched base alternatives before fitting the joint elites.
            refine_count = max(3, self.count // 3)
            batch_sizes = (self.count - refine_count, refine_count)
        for _ in range(self.iterations):
            if self.sumo_sampling:
                ramp = np.linspace(3.5 / len(self.nodes), 3.5, len(self.nodes))
                sigma = np.clip(sigma * ramp[:, None], .12, .8)
            population, scores = [], []
            parent = None
            for count in batch_sizes:
                if parent is None:
                    options = {"proposals": propose(self)} if propose is not None else {}
                    knots = sample(mean[:, sampled], sigma[:, sampled], incumbent[:, sampled],
                                   self.previous[sampled], self.scale[sampled], count, self.rng,
                                   self.nodes, stance_rate, force_rng=self.force_rng, **options)
                    if stance is not None:
                        knots = np.concatenate((knots, stance), axis=2)
                else:
                    knots = refine_base(parent, sigma[1, 4:], stance_rate, self.previous[4:],
                                        self.scale[4:], count, self.rng, self.nodes[1] - self.nodes[0])
                commands, forecasts, elapsed = self._evaluate(snapshot, knots)
                costs = score(forecasts)
                costs[~np.isfinite(costs)] = np.inf
                rollout_s += elapsed
                population.append(knots)
                scores.append(costs)
                index = np.argmin(costs)
                if not np.isfinite(costs[index]):
                    continue
                parent = knots[index]
                if costs[index] < best_cost:
                    best_cost = costs[index]
                    incumbent = knots[index].copy()
                    prefix = commands[index, :PREFIX].copy()
                    fields = ("qpos", "action", "policy_command")
                    if "gripper_on_cone_N" in forecasts:
                        fields += ("qvel", "gripper_on_cone_N", "ground_on_cone_N", "gripper_touch")
                    else:
                        fields += ("gripper_on_box_N",)
                    prediction = {k: forecasts[k][index, :PREFIX].copy() for k in fields}
                    if capture_path:
                        roots = np.vstack((root, forecasts["qpos"][index, :-1, :7]))
                        effective = world_ee_target(roots, forecasts["policy_command"][index, :, 3:6])
                        paths = (commands[index, :, :3].copy(), effective,
                                 forecasts["ee_world"][index].copy())
            costs = np.concatenate(scores)
            valid = np.isfinite(costs).sum()
            valid_count += valid
            if valid:
                mean, sigma, _ = fit(np.concatenate(population), costs, 3 if self.sumo_sampling else 2)
        if np.isfinite(best_cost):
            last = prefix[-1].astype(float)
            if self.force_rng is not None:
                last = np.r_[last[:6], last[6:] @ self.push_axis]
            if self.yaw_control and not self.planar:
                last = np.delete(last, 4)
            if self.rotation is not None:
                last[:2] = last[:2] @ self.rotation
            self.previous = (last - self.center) / self.scale
            shifted = self.nodes + PREFIX * DT
            self.mean = interpolate(mean, shifted, self.nodes)
            self.winner = interpolate(incumbent, shifted, self.nodes)
            self.sigma = interpolate(sigma, shifted, self.nodes)
            selected = self.center + self.scale * incumbent
            if self.rotation is not None:
                selected[:, :2] = selected[:, :2] @ self.rotation.T
        else:
            prefix, prediction = None, {}
            selected = np.full_like(self.mean, np.nan)
        plan = dict(cost=best_cost, knots_m=selected[:, :3], valid_candidates=valid_count,
                    rollout_s=rollout_s)
        if capture_path:
            plan["view_paths"] = paths
        if self.mobile:
            plan["knots_vx_m_s"] = selected[:, 3]
        if self.planar:
            plan["knots_vy_m_s"] = selected[:, 4]
        if self.yaw_control:
            plan["knots_yaw_rad_s"] = selected[:, 5 if self.planar else 4]
        if self.force_rng is not None:
            plan["knots_push_N"] = selected[:, 6]
        return prefix, prediction, plan


def latency(seconds):
    ms = np.asarray(seconds) * 1000
    return dict(p50_ms=float(np.median(ms)), p95_ms=float(np.percentile(ms, 95)),
                max_ms=float(ms.max()), misses_100ms=int(np.sum(ms > 100)))


def settling_times(errors, speeds, initial_error):
    """First qualifying prefix end (control seconds), observed until precision termination."""
    radii = {"coarse": min(.05, initial_error / 2), "precision": RADIUS, "high_precision": .01}
    times = dict.fromkeys(radii)
    if len(errors) >= SETTLE_TICKS:
        window = np.lib.stride_tricks.sliding_window_view
        peak_error = window(errors, SETTLE_TICKS)[::PREFIX].max(axis=1)
        peak_speed = window(speeds[SETTLE_TICKS-PREFIX:], PREFIX)[::PREFIX].max(axis=1)
        for name, radius in radii.items():
            if radius == 0.:
                continue  # A zero-length request cannot demonstrate coarse pushing progress.
            hits = np.flatnonzero((peak_error <= radius) & (peak_speed <= SETTLE_SPEED))
            if len(hits):
                times[name] = float((SETTLE_TICKS + hits[0] * PREFIX) * DT)
    return times


def run(args, *, scenario=None, horizon=.5, objective=None, moving_bounds=False,
        planar=False, sumo_sampling=False, live=None, pool=None, display=None, quiet=False,
        base_control=None):
    start = perf_counter()
    settings = json.loads((scenario / "scenario.json").read_text()) if scenario else None
    randomized_mobile = scenario is not None and args.controller == "mobile"
    objective = objective or ("bounded" if randomized_mobile else "terminal")
    if randomized_mobile and base_control is None and not planar:
        base_control = "full"
    seconds = args.seconds
    if seconds is None:
        seconds = episode_seconds(settings.get("difficulty") if settings else None)
    scene = scenario / "scene.mjb" if scenario else args.scene
    if live is None:
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
        raise InvalidSpawn("Robot touched the box during scenario warmup")
    anchor = warmup_target(warmup)
    ee_force = getattr(args, "ee_force", "off")
    planner = None
    if args.controller in ("mpc", "mobile"):
        planner = BoxPlanner(live, box_start, anchor, args.candidates, args.iterations,
                             args.workers, args.seed, mobile=args.controller == "mobile",
                             horizon=horizon, objective=objective, moving_bounds=moving_bounds,
                             planar=planar, sumo_sampling=sumo_sampling, pool=pool,
                             base_control=base_control, ee_force=ee_force,
                             push_direction=goal - box_start[:2],
                             workspace_yaw=settings["robot_yaw_rad"] if settings and
                             "difficulty" in settings else None)
    traces = [warmup]
    plans, cycles, requests = [], [], []
    in_goal = 0
    status = "time_limit"
    if display:
        display.play(warmup, goal, None, None)
    control_start = perf_counter()
    for update in range(int(seconds / (DT * PREFIX) + 1e-9)):
        cycle_start = perf_counter()
        paths = planning_s = None
        if planner:
            q = traces[-1]["qpos"][-1]
            finger = traces[-1]["ee_world"][-1]
            gap = np.linalg.norm(finger - q[-7:-4]) if objective == "hybrid" else None
            commands, predicted, plan = planner.plan(live, q[-7:-5], goal, finger_gap=gap,
                                                     capture_path=display is not None, root=q[:7])
            paths = plan.pop("view_paths", None)
            plan["planning_s"] = planning_s = perf_counter() - cycle_start
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
        executed = live.run(commands, matched_position=ee_force == "matched")
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
        if display:
            display.play(executed, goal, paths, planning_s)
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
    length = np.linalg.norm(axis)
    axis /= max(length, 1e-12)
    errors = np.linalg.norm(qpos[WARMUP_TICKS:, -7:-5] - goal, axis=1)
    speeds = np.linalg.norm(trace["qvel"][WARMUP_TICKS:, -6:-4], axis=1)
    times = settling_times(errors, speeds, length)
    summary = dict(controller=args.controller, status=status, goal_xy_m=goal.tolist(),
                   tolerance_m=RADIUS, required_hold_s=SETTLE_TICKS*DT, settle_speed_m_s=SETTLE_SPEED,
                   settle_speed_window_s=PREFIX*DT, seed=args.seed, scene=str(scene),
                   warmup_ee_yaw_frame_m=args.warmup_ee or [.20, 0., .10],
                   box_start_xy_m=box_start[:2].tolist(), displacement_xy_m=displacement.tolist(),
                   lateral_drift_m=float(displacement @ [-axis[1], axis[0]]),
                   final_error_m=float(np.linalg.norm(qpos[-1, -7:-5] - goal)), goal_hold_s=in_goal*DT,
                   control_limit_s=seconds, settle_times_s=times,
                   progress_pct=float(100 * (displacement @ axis) / length) if length else None,
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
                       horizon_s=horizon, objective=objective, moving_bounds=planner.moving_bounds,
                       planar=planner.planar, base_control=planner.base_control,
                       sumo_sampling=sumo_sampling,
                       planning=latency([p["planning_s"] for p in plans]),
                       rollout=latency([p["rollout_s"] for p in plans]),
                       prefix_max_error=max((float(p["prefix_max_error"]) for p in plans
                                             if p["valid_candidates"]), default=None))
    if ee_force != "off":
        summary["ee_force"] = ee_force
    if settings is not None:
        summary["scenario"] = settings
    if not quiet:
        print(json.dumps(summary, indent=2))
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        np.savez_compressed(args.out / "episode.npz", **trace, goal_xy=goal,
                            time_s=(np.arange(len(qpos)) + 1)*DT, cycle_s=cycles)
        if plans:
            width = 6 if planner.yaw_control else 3 + int(planner.mobile) + int(planner.planar)
            if ee_force != "off":
                width = 8
            prefixes = np.asarray(requests).reshape(-1, PREFIX, width)
            fields = {k: [p[k] for p in plans] for k in plans[0]}
            if planner.mobile:
                fields["requested_vx_m_s"] = prefixes[..., 3]
            if planner.planar:
                fields["requested_vy_m_s"] = prefixes[..., 4]
            if planner.yaw_control:
                fields["requested_yaw_rad_s"] = prefixes[..., 5]
            if ee_force != "off":
                fields["requested_force_xy_N"] = prefixes[..., 6:]
            np.savez_compressed(args.out / "plans.npz", requested_prefix_m=prefixes[..., :3],
                                **fields)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=("mpc", "mobile", "hold", "scripted"),
                        help="Defaults to mobile for randomized scenes and mpc for the archived demo")
    parser.add_argument("--warmup-ee", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        help="Initial Cartesian EE request relative to the yaw-frame sphere origin (m)")
    targets = parser.add_mutually_exclusive_group()
    targets.add_argument("--goal", type=float, nargs=2, default=(1.16, 0.), metavar=("X", "Y"),
                         help="Absolute target for the box center in world metres")
    targets.add_argument("--scenario", type=Path, help="Generated scene directory with a relative goal")
    parser.add_argument("--seconds", type=float, help="Override the tier's control duration after warmup")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=min(8, len(os.sched_getaffinity(0))))
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--horizon", type=float, choices=(.5, 1., 1.5), default=.5)
    parser.add_argument("--objective", choices=("terminal", "running", "path", "bounded", "speed",
                                                "settle", "proximity", "hybrid"))
    parser.add_argument("--sumo-sampling", action="store_true", help="Use three elites and a 3.5 noise ramp")
    parser.add_argument("--ee-force", choices=("off", "conditioned", "matched"), default="off",
                        help="Optimize signed push-axis force or its reachable position equivalent")
    parser.add_argument("--moving-bounds", action="store_true", help="Translate XY bounds with the live box")
    base = parser.add_mutually_exclusive_group()
    base.add_argument("--base", choices=BASE_CONTROLS)
    base.add_argument("--planar", action="store_const", dest="base", const="lateral",
                      help="Sample body-lateral speed together with forward speed")
    parser.add_argument("--candidates", type=int)
    parser.add_argument("--scene", type=Path, default=Path("artifacts/spot_box.mjb"))
    parser.add_argument("--policy", type=Path, default=Path("artifacts/encoder_actor.onnx"))
    parser.add_argument("--out", type=Path, help="Directory for executed trace, summary and MPC plans")
    args = parser.parse_args()
    args.controller = args.controller or ("mobile" if args.scenario else "mpc")
    randomized = args.scenario is not None and args.controller == "mobile"
    if args.candidates is None:
        args.candidates = 24 if randomized else 32
    if (not np.isfinite(args.goal).all()
            or args.seconds is not None and (not np.isfinite(args.seconds) or args.seconds < .1)
            or (args.warmup_ee is not None and not np.isfinite(args.warmup_ee).all())
            or args.candidates < 3 or min(args.iterations, args.workers) < 1):
        parser.error("Expected finite goal, at least 0.1 s, three candidates and positive counts")
    if args.base not in (None, "forward") and args.controller != "mobile":
        parser.error("Extra base axes require --controller mobile")
    if args.ee_force != "off" and (args.controller != "mobile" or args.base not in (None, "full")):
        parser.error("Use --controller mobile --base full with --ee-force")
    if args.ee_force != "off":
        args.base = "full"
    run(args, scenario=args.scenario, horizon=args.horizon, objective=args.objective,
        moving_bounds=args.moving_bounds, base_control=args.base, sumo_sampling=args.sumo_sampling)
