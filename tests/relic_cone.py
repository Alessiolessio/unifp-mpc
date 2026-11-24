"""Two short reference runs using SUMO's registered ReLIC controller and live backend."""

import argparse
import json
from time import perf_counter
from pathlib import Path
from tempfile import TemporaryDirectory

import judo.tasks.base as base
import mujoco
import numpy as np
from judo.app.structs import MujocoState
from judo.simulation import get_simulation_backend
from sumo.controller import make_controller


def run(label, pose=None, *, seed=43, paired=False):
    task_name = "spot_cone_upright"
    sim = get_simulation_backend("mujoco_hierarchical")(init_task=task_name)
    ctrl = make_controller(task_name, "cem")
    task = sim.task
    np.random.seed(0)
    task.reset()
    data = task.data
    if pose is not None:
        data.qpos[:2] = pose[:2]
        data.qpos[3:7] = pose[3:7]
        data.qpos[-7:] = pose[-7:]
        mujoco.mj_forward(task.model, data)
    warm_ticks = round(1/task.dt) if paired else 0
    cone = task.model.body("traffic_cone").id
    for _ in range(warm_ticks):
        sim.step(task.optimizer_warm_start())
        for contact in data.contact:
            bodies = task.model.geom_bodyid[contact.geom]
            if cone in bodies and 0 not in bodies and contact.efc_address >= 0:
                raise ValueError("Robot touched cone during warmup")
    if paired and (np.linalg.norm(data.qvel[-3:]) > .05 or np.linalg.norm(data.qvel[-6:-3]) > .02):
        raise ValueError("Expected settled cone after neutral warmup")
    ctrl.reset()
    np.random.seed(seed)
    elapsed = []
    poses, hands = [data.qpos.copy()], []
    next_plan = warm_ticks * task.dt
    print(
        label,
        type(ctrl.rollout_backend).__name__,
        ctrl.controller_cfg,
        ctrl.optimizer_cfg,
        flush=True,
    )
    for tick in range(round(12 / task.dt)):
        t = (tick + warm_ticks) * task.dt
        if t + 1e-9 >= next_plan:
            before = perf_counter()
            ctrl.update_states(
                MujocoState(
                    t,
                    data.qpos.copy(),
                    data.qvel.copy(),
                    data.mocap_pos.copy(),
                    data.mocap_quat.copy(),
                    task.get_sim_metadata(),
                )
            )
            ctrl.update_action()
            elapsed.append(perf_counter()-before)
            next_plan += 1 / ctrl.controller_cfg.control_freq
        sim.step(ctrl.action(t))
        poses.append(data.qpos.copy())
        hands.append(data.site("site_arm_link_fngr").xpos.copy())
    q = np.array(poses)
    angle = np.rad2deg(np.arccos(np.clip(1 - 2 * (q[:, -3] ** 2 + q[:, -2] ** 2), -1, 1)))
    print(label, "initial/min/final tilt", angle[0], angle.min(), angle[-1], flush=True)
    dest = Path("artifacts/uncertainty/sumo" if paired else "artifacts/ee_study")
    dest.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(dest/f"relic_{label}.npz", qpos=q, angle=angle, finger=hands,
                        planning_s=elapsed, dt=task.dt)
    if paired:
        ms = 1000*np.array(elapsed)
        crossed = np.flatnonzero(angle <= np.rad2deg(np.arccos(.9)))
        period = 1000/ctrl.controller_cfg.control_freq
        result = dict(case=label, seed=seed, dt=task.dt, horizon=ctrl.horizon,
                      draws=ctrl.optimizer_cfg.num_rollouts, steps=int(ctrl.num_timesteps),
                      nodes=ctrl.optimizer_cfg.num_nodes, period_ms=period,
                      initial_cone=q[0,-7:].tolist(), settled_root=q[0,:7].tolist(),
                      first_upright_s=float(task.dt*crossed[0]) if len(crossed) else None,
                      final_deg=angle[-1], min_deg=angle.min(), p50_ms=np.median(ms),
                      p95_ms=np.percentile(ms,95), max_ms=ms.max(), updates=len(ms),
                      period_misses=int(sum(ms>period)), misses_100ms=int(sum(ms>100)))
        (dest/f"{label}.json").write_text(json.dumps(result, indent=2))
        print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired-case", type=int, choices=[0, 2])
    parser.add_argument("--equal-work", action="store_true")
    args = parser.parse_args()
    Path("artifacts/ee_study").mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as tmp:
        # Judo acquires a lock even for a cache hit. Keep that lock outside the reference install.
        (Path(tmp) / "meshes").symlink_to(base.MODEL_PATH / "meshes", target_is_directory=True)
        base.MODEL_PATH = Path(tmp)
        if args.paired_case is not None:
            from judo.config import set_config_overrides
            from judo.controller.controller import ControllerConfig
            from judo.optimizers.cem import CrossEntropyMethodConfig

            folder = Path("artifacts/uncertainty/cone")/f"{args.paired_case:02d}"
            scene = mujoco.MjModel.from_binary_path(str(folder/"nominal"/"scene.mjb"))
            if args.equal_work:
                set_config_overrides("spot_cone_upright", ControllerConfig, {"control_freq": 10.})
                set_config_overrides("spot_cone_upright", CrossEntropyMethodConfig, {"num_rollouts": 8})
            label = f"{args.paired_case}-" + ("equal" if args.equal_work else "native")
            seed = json.loads((folder/"off"/"summary.json").read_text())["optimizer_seed"]
            run(label, scene.key("spawn").qpos, seed=seed, paired=True)
            raise SystemExit
        run("natural")
        q = np.load("artifacts/cone_clearance/general-buffer/82/episode.npz")["qpos"][49].copy()
        # Match horizontal root pose and settled cone, keeping ReLIC's own reset joints/height.
        w, x, y, z = q[3:7]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        q[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        run("away82", q)
