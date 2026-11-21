"""Replay archived MPC to a contact state, then compare closed-loop continuations."""

import argparse
import copy
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from box_mpc import (
    BASE_CONTROLS, BoxPlanner, RADIUS, SETTLE_SPEED, SETTLE_TICKS, latency, warmup_target,
)
from cem import DT, PREFIX
from run_world import command_schedule
from unifp_native import Rollouts, World

CASES = {
    "hard9": ("artifacts/viewer_study/hard9/bounded", (40, 55)),
    "long14": ("artifacts/robust_box/held_selected/14", (40, 65)),
}


def make_branch(live, pool, ref, box, anchor, settings, mode, objective):
    b = BoxPlanner(live, box, anchor, 24, 1, 8, 0, mobile=True, pool=pool,
                   objective=objective, workspace_yaw=settings.get("robot_yaw_rad")
                   if "difficulty" in settings else None, base_control=mode)
    # Preserve the forward controller's fitted CEM distribution, incumbent, request and RNG.
    # Newly available base axes start at zero command with their original exploration variance.
    for key in ("mean", "sigma", "winner", "center", "previous"):
        getattr(b, key)[..., :4] = getattr(ref, key)
    b.rng.bit_generator.state = copy.deepcopy(ref.rng.bit_generator.state)
    return b


def continuation(live, planner, initial, goal, start, limit, root, archive=None):
    axis = goal - planner.start_xy
    axis /= np.linalg.norm(axis)
    side = np.array([-axis[1], axis[0]])
    xy = initial.copy()
    hold = 0
    timings, touched, corrective, cross = [], [], [], []
    valid = 0
    status = "time_limit"
    for i in range(start, limit):
        t = perf_counter()
        prefix, prediction, plan = planner.plan(live, xy, goal, root=root)
        timings.append(perf_counter() - t)
        valid += int(plan["valid_candidates"])
        if prefix is None:
            status = "no_valid_plan"
            break
        trace = live.run(prefix)
        for k, expected in prediction.items():
            np.testing.assert_array_equal(trace[k], expected)
        q = trace["qpos"]
        if archive is not None and 50 + (i + 1)*PREFIX <= len(archive):
            np.testing.assert_array_equal(q, archive[50 + i*PREFIX:50 + (i+1)*PREFIX])
        line = goal - np.vstack((xy, q[:-1, -7:-5]))
        useful = np.sum(trace["gripper_on_box_N"][:, :2] * line, axis=1) > 0.
        contact = trace["contact_duty"] > 0
        dist = np.linalg.norm(goal - q[:, -7:-5], axis=1)
        touched.extend(contact)
        corrective.extend(contact & useful & (dist < np.linalg.norm(line, axis=1)))
        cross.extend((q[:, -7:-5] - planner.start_xy) @ side)
        xy = q[-1, -7:-5].copy()
        root = q[-1, :7]
        for err in dist:
            hold = hold + 1 if err <= RADIUS else 0
        peak_speed = np.linalg.norm(trace["qvel"][:, -6:-4], axis=1).max()
        if hold >= SETTLE_TICKS and peak_speed <= SETTLE_SPEED:
            status = "success"
            break
    hits = np.flatnonzero(corrective)
    return dict(status=status, error_mm=float(1000*np.linalg.norm(goal-xy)),
                contact_ticks=int(np.sum(touched)), corrective_ticks=len(hits),
                useful_contact_s=float((hits[0]+1)*DT) if len(hits) else None,
                start_cross_mm=1000*(initial-planner.start_xy)@side,
                end_cross_mm=1000*(xy-planner.start_xy)@side,
                peak_cross_mm=1000*max(np.abs(cross), default=0.),
                control_s=len(touched)*DT, updates=len(timings), valid_candidates=valid,
                planning=latency(timings))


def compare(case, modes, objective, limit):
    folder, checkpoints = CASES[case]
    folder = Path(folder)
    s = json.loads((folder / "summary.json").read_text())
    settings = s["scenario"]
    live = World(s["scene"], "artifacts/encoder_actor.onnx")
    pool = Rollouts(live, 8)
    warm = live.run(command_schedule("push")[:50])
    anchor = warmup_target(warm)
    box = warm["qpos"][-1, -7:-4]
    goal = np.array(s["goal_xy_m"])
    ref = BoxPlanner(live, box, anchor, 24, 1, 8, s["seed"], mobile=True, pool=pool,
                     objective="bounded", workspace_yaw=settings.get("robot_yaw_rad")
                     if "difficulty" in settings else None)
    xy = box[:2]
    root = warm["qpos"][-1, :7]
    with np.load(folder / "episode.npz") as old:
        archived = old["qpos"]
    for i in range(max(checkpoints)+1):
        if i in checkpoints:
            assert np.linalg.norm(xy - goal) > RADIUS  # No existing goal-hold counter at these cuts.
            state = live.snapshot()
            for mode in modes:
                live.restore(state)
                branch = make_branch(live, pool, ref, box, anchor, settings, mode, objective)
                oracle = archived if mode == "forward" and objective == "bounded" else None
                result = continuation(live, branch, xy, goal, i, limit, root, oracle)
                print(json.dumps(dict(case=case, start_s=i/10, limit_s=limit/10, mode=mode,
                                      objective=objective, **result)), flush=True)
            live.restore(state)
        if i < max(checkpoints):
            prefix, _, _ = ref.plan(live, xy, goal)
            q = live.run(prefix)["qpos"]
            np.testing.assert_array_equal(q, archived[50+i*PREFIX:50+(i+1)*PREFIX])
            xy = q[-1, -7:-5]
            root = q[-1, :7]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    p.add_argument("--modes", nargs="+", choices=BASE_CONTROLS, default=BASE_CONTROLS)
    p.add_argument("--objective", choices=("bounded", "proximity"), default="bounded")
    p.add_argument("--seconds", type=int, default=9, help="Absolute control time limit after warmup")
    args = p.parse_args()
    if args.seconds < 7:
        p.error("Time limit must include the recovery checkpoints")
    for case in args.cases:
        compare(case, args.modes, args.objective, 10 * args.seconds)
