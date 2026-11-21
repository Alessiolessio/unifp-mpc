"""Continuously run seeded UniFP box MPC episodes in the passive MuJoCo viewer."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from itertools import count
import json
import os
from pathlib import Path
import signal
import tempfile
import threading
from time import perf_counter, sleep

import mujoco
import numpy as np

from box_mpc import BASE_CONTROLS, RADIUS, run, yaw_of
from box_scenario import DIFFICULTIES, InvalidSpawn, episode_seconds, generate, scene_template
from cem import DT
from render_box import mark_centers
from unifp_native import Rollouts, TrajectoryFailure, World


class WindowClosed(Exception):
    pass


@contextmanager
def open_window(model, data):
    import glfw
    from mujoco import viewer

    # On Linux launch_passive waits indefinitely for its thread if GLFW cannot initialize.
    if not glfw.init():
        raise SystemExit("Cannot open a GLFW display. Use a graphical session or --headless.")
    before = set(threading.enumerate())
    window = viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
    launched = set(threading.enumerate()) - before
    try:
        yield window
    finally:
        # Handle.close() requests exit asynchronously. Finish GL teardown before interpreter exit.
        window.close()
        for thread in launched:
            thread.join()


def draw_path(scene, points, color):
    for a, b in zip(points[:-1], points[1:]):
        if np.array_equal(a, b):
            continue
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), a,
                            np.eye(3).ravel(), color)
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 3, a, b)
        scene.ngeom += 1


class Playback:
    """Present one executed prefix while the controller computes the next one."""

    def __init__(self, window, data, executor, paced):
        self.window = window
        self.data = data
        self.executor = executor
        self.paced = paced
        self.pending = None

    def check(self):
        if not self.window.is_running():
            raise WindowClosed()

    def wait(self):
        if self.pending is not None:
            pending, self.pending = self.pending, None
            pending.result()
        self.check()

    def begin(self, label, totals):
        self.wait()
        self.label = label
        self.totals = totals
        self.frames = self.ticks = 0
        self.first_frame = None
        self.finger = None
        self.error = None

    def play(self, trace, goal, paths, planning):
        self.wait()
        self.pending = self.executor.submit(self._present, trace, goal, paths, planning)

    def hud(self, status):
        self.window.set_texts((mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                               f"{self.label}\n{self.totals}\n{status}",
                               "Cyan   requested EE\nAmber   predicted finger\nGreen   15 mm goal"))

    def _present(self, trace, goal, paths, planning):
        self.check()
        w, scn = self.window, self.window.user_scn
        qpos = trace["qpos"]
        start = perf_counter()
        if self.first_frame is None:
            self.first_frame = start
        with w.lock():
            scn.ngeom = 0
            mark_centers(scn, qpos[0, -7:-4], goal, RADIUS)
            box_marker = scn.ngeom - 1
            if paths is not None:
                targets, fingers = paths
                draw_path(scn, targets, [0., .9, 1., 1.])
                mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                    [.008]*3, targets[-1], np.eye(3).ravel(), [0., .9, 1., 1.])
                scn.ngeom += 1
                draw_path(scn, np.vstack((self.finger, fingers)), [1., .57, .03, 1.])
            finger_marker = scn.ngeom
            mujoco.mjv_initGeom(scn.geoms[finger_marker], mujoco.mjtGeom.mjGEOM_SPHERE,
                                [.009]*3, trace["ee_world"][0], np.eye(3).ravel(),
                                [1., .57, .03, 1.])
            scn.ngeom += 1
            if self.finger is None:
                # Frame the new scene once. The user owns the camera during the episode.
                w.cam.lookat[:] = [*(.5 * (qpos[0, :2] + goal)), .42]
                w.cam.distance = max(2.3, 1.5 + np.linalg.norm(goal - qpos[0, :2]) * .7)
                w.cam.azimuth = 105 + np.rad2deg(yaw_of(qpos[0, 3:7]))
        i = 0 if self.paced else len(qpos) - 1
        while i < len(qpos):
            self.check()
            with w.lock():
                self.data.qpos[:] = qpos[i]
                self.data.qvel[:] = trace["qvel"][i]
                self.data.time = (self.ticks + i + 1) * DT
                scn.geoms[box_marker].pos[:] = qpos[i, -7:-4] + [0, 0, .261]
                scn.geoms[finger_marker].pos[:] = trace["ee_world"][i]
            self.error = np.linalg.norm(qpos[i, -7:-5] - goal)
            if i == 0 or not self.paced:
                timing = "Warmup" if planning is None else f"Plan {planning*1000:.1f} ms"
                self.hud(f"{timing}    error {self.error*100:.1f} cm")
            w.sync(state_only=True)
            self.frames += 1
            if i == len(qpos) - 1:
                break
            # Drop late display samples only. The native controller always executes all ticks.
            i = min(len(qpos) - 1, max(i + 2, int((perf_counter() - start) / DT)))
            sleep(max(0., start + i * DT - perf_counter()))
        if self.paced:
            sleep(max(0., start + len(qpos) * DT - perf_counter()))
        self.ticks += len(qpos)
        self.finger = trace["ee_world"][-1]

    def finish(self, status, totals, pause):
        self.wait()
        self.totals = totals
        error = "" if self.error is None else f"    error {self.error*100:.1f} cm"
        self.hud(status + error)
        elapsed = perf_counter() - self.first_frame if self.first_frame is not None else 0.
        end = perf_counter() + pause
        while perf_counter() < end:
            self.check()
            sleep(min(.05, max(0., end - perf_counter())))
        return dict(pose_frames=self.frames, playback_wall_s=elapsed,
                    pose_fps=self.frames / elapsed if elapsed else 0.)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--difficulty", choices=["cycle", *DIFFICULTIES], default="cycle")
    p.add_argument("--scenario-seed", type=int, default=0, help="First seed, incremented per episode")
    p.add_argument("--cem-seed", type=int, default=17, help="Independent optimizer seed for each reset")
    p.add_argument("--episodes", type=int, help="Stop after this many episodes (default: continuous)")
    p.add_argument("--seconds", type=float, help="Override tier duration (9, 15 or 20 s after warmup)")
    p.add_argument("--pause", type=float, default=1., help="Wall seconds to show the outcome")
    p.add_argument("--unpaced", action="store_true", help="Display the last executed tick of each prefix")
    p.add_argument("--headless", action="store_true", help="Same continuous controller without playback")
    p.add_argument("--base", choices=BASE_CONTROLS, default="full",
                   help="Use stance for geometric vy/yaw or guided for CEM refinement of those rates")
    p.add_argument("--objective", choices=("bounded", "proximity", "hybrid"), default="bounded")
    p.add_argument("--horizon", type=float, choices=(.5, 1., 1.5), default=.5)
    p.add_argument("--candidates", type=int, default=24)
    p.add_argument("--iterations", type=int, default=1)
    p.add_argument("--sumo-sampling", action="store_true", help="Use three elites and a 3.5 noise ramp")
    p.add_argument("--workers", type=int, default=min(8, len(os.sched_getaffinity(0))))
    p.add_argument("--log", type=Path, default=Path("artifacts/live_box/episodes.jsonl"))
    p.set_defaults(controller="mobile", warmup_ee=None, out=None,
                   policy=Path("artifacts/encoder_actor.onnx"))
    args = p.parse_args()
    if (min(args.workers, args.iterations) < 1 or args.candidates < 3
            or min(args.scenario_seed, args.cem_seed) < 0
            or args.seconds is not None and (not np.isfinite(args.seconds) or args.seconds < .1)
            or not np.isfinite(args.pause) or args.pause < 0
            or args.episodes is not None and args.episodes < 1):
        p.error("Invalid episode, timing, seed or planner counts")
    args.seed = args.cem_seed
    args.log.parent.mkdir(parents=True, exist_ok=True)
    model = scene_template()
    pool = display = None
    successes = failures = coarse = high = 0
    totals = "0 episodes"
    with ExitStack() as stack:
        tmp = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        log = stack.enter_context(args.log.open("a", buffering=1))
        if not args.headless:
            visual = mujoco.MjModel.from_binary_path("artifacts/spot_box.mjb")
            visual.geom("ground").group = 0
            data = mujoco.MjData(visual)
            window = stack.enter_context(open_window(visual, data))
            executor = stack.enter_context(ThreadPoolExecutor(max_workers=1))
            with window.lock():
                window.cam.elevation = -28
                for flag in (mujoco.mjtRndFlag.mjRND_SHADOW, mujoco.mjtRndFlag.mjRND_REFLECTION):
                    window.user_scn.flags[flag] = 0
            display = Playback(window, data, executor, not args.unpaced)

        for episode in count():
            seed = args.scenario_seed + episode
            tier = args.difficulty
            if tier == "cycle":
                tier = DIFFICULTIES[episode % len(DIFFICULTIES)]
            label = f"{tier.title()} ({args.base})    scene {seed}    CEM {args.seed}"
            result = dict(scenario_seed=seed, cem_seed=args.seed, difficulty=tier, horizon_s=args.horizon,
                          objective=args.objective, candidates=args.candidates, workers=args.workers,
                          seconds=args.seconds or episode_seconds(tier), base_control=args.base,
                          iterations=args.iterations, sumo_sampling=args.sumo_sampling)
            started = perf_counter()
            try:
                if display:
                    display.begin(label, totals)
                result["scenario"] = generate(tmp, seed, difficulty=tier, model=model)
                scene = str(tmp / "scene.mjb")
                if pool is None:
                    live = World(scene, str(args.policy))
                    pool = Rollouts(live, args.workers)
                else:
                    live = pool.new_episode(scene)
                s = run(args, scenario=tmp, live=live, pool=pool, display=display,
                        horizon=args.horizon, objective=args.objective, base_control=args.base,
                        sumo_sampling=args.sumo_sampling, quiet=True)
                for key in ("status", "goal_xy_m", "final_error_m", "planning", "mpc_updates",
                            "simulated_s", "prefix_max_error", "gripper_contact_ticks", "progress_pct",
                            "settle_times_s", "max_abs_roll_pitch_deg", "other_robot_box_impulse_Ns"):
                    result[key] = s[key]
                if display:
                    display.wait()
            except (InvalidSpawn, TrajectoryFailure) as exc:
                result.update(status="invalid_spawn" if isinstance(exc, InvalidSpawn) else "live_failure",
                              detail=str(exc))
            except (WindowClosed, KeyboardInterrupt):
                result["status"] = "interrupted"
                print(json.dumps(result), file=log)
                break

            result["episode_wall_s"] = perf_counter() - started
            successes += int(result["status"] == "success")
            failures += int(result["status"] != "success")
            settled = result.get("settle_times_s", {})
            coarse += int(settled.get("coarse") is not None)
            high += int(settled.get("high_precision") is not None)
            totals = (f"{episode+1} episodes   settled at 5 cm / 15 mm / 10 mm"
                      f"   {coarse} / {successes} / {high}")
            result.update(successes=successes, failures=failures, coarse_successes=coarse,
                          high_precision_successes=high)
            status = result["status"]
            if display:
                try:
                    result.update(display.finish(status, totals, args.pause))
                except (WindowClosed, KeyboardInterrupt):
                    print(json.dumps(result), file=log)
                    break
            print(json.dumps(result), file=log)
            error = result.get("final_error_m")
            p95 = result.get("planning", {}).get("p95_ms")
            metrics = f"    {100*error:.1f} cm    plan p95 {p95:.1f} ms" if p95 is not None else ""
            print(f"{label}    {status}{metrics}    {totals}", flush=True)
            if args.episodes == episode + 1:
                break


if __name__ == "__main__":
    def terminate(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    try:
        main()
    except KeyboardInterrupt:
        pass
