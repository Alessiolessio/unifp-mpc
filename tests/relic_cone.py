"""Two short reference runs using SUMO's registered ReLIC controller and live backend."""

from pathlib import Path
from tempfile import TemporaryDirectory

import judo.tasks.base as base
import mujoco
import numpy as np
from judo.app.structs import MujocoState
from judo.simulation import get_simulation_backend
from sumo.controller import make_controller


def run(label, pose=None):
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
    ctrl.reset()
    np.random.seed(43)
    poses, hands = [data.qpos.copy()], []
    next_plan = 0.0
    print(
        label,
        type(ctrl.rollout_backend).__name__,
        ctrl.controller_cfg,
        ctrl.optimizer_cfg,
        flush=True,
    )
    for tick in range(round(12 / task.dt)):
        t = tick * task.dt
        if t + 1e-9 >= next_plan:
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
            next_plan += 1 / ctrl.controller_cfg.control_freq
        sim.step(ctrl.action(t))
        poses.append(data.qpos.copy())
        hands.append(data.site("site_arm_link_fngr").xpos.copy())
    q = np.array(poses)
    angle = np.rad2deg(np.arccos(np.clip(1 - 2 * (q[:, -3] ** 2 + q[:, -2] ** 2), -1, 1)))
    print(label, "initial/min/final tilt", angle[0], angle.min(), angle[-1], flush=True)
    np.savez_compressed(
        f"artifacts/ee_study/relic_{label}.npz", qpos=q, angle=angle, finger=hands, dt=task.dt
    )


if __name__ == "__main__":
    Path("artifacts/ee_study").mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as tmp:
        # Judo acquires a lock even for a cache hit. Keep that lock outside the reference install.
        (Path(tmp) / "meshes").symlink_to(base.MODEL_PATH / "meshes", target_is_directory=True)
        base.MODEL_PATH = Path(tmp)
        run("natural")
        q = np.load("artifacts/cone_clearance/general-buffer/82/episode.npz")["qpos"][49].copy()
        # Match horizontal root pose and settled cone, keeping ReLIC's own reset joints/height.
        w, x, y, z = q[3:7]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        q[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        run("away82", q)
