"""Run the registered ReLIC planner against a local shared manipulation fixture."""
import argparse
import ctypes
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import judo.tasks.base as base
import mujoco
import numpy as np
from judo.app.structs import MujocoState
from judo.config import set_config_overrides
from judo.controller.controller import ControllerConfig
from judo.optimizers.cem import CrossEntropyMethodConfig
from judo.simulation import get_simulation_backend
from sumo.controller import make_controller
from sumo.tasks.spot.spot_box_push import SpotBoxPush

from physical_score import score, tilt


def shared_box(task, source):
    geom = source.geom("box_collision")
    body = source.body("box_body")
    dst = task.spec.body("box_body")
    dst.mass, dst.inertia = body.mass.item(), body.inertia
    for name in ("box_visual", "box_collision"):
        task.spec.geom(name).size = geom.size
    g = task.spec.geom("box_collision")
    g.priority, g.friction = geom.priority.item(), geom.friction
    g.solref, g.solimp = geom.solref, geom.solimp
    pair = source.pair("box_ground")
    task.spec.add_pair(name="box_ground", geomname1="ground", geomname2="box_collision",
                       condim=pair.dim.item(), solref=pair.solref, solimp=pair.solimp,
                       friction=pair.friction)


def verify(source, model, body):
    for key in ("mass", "inertia", "ipos", "iquat"):
        np.testing.assert_allclose(getattr(model.body(body), key), getattr(source.body(body), key))
    if body == "box_body":
        for key in ("dim", "friction", "solimp", "solref"):
            np.testing.assert_allclose(getattr(model.pair("box_ground"), key),
                                       getattr(source.pair("box_ground"), key))
    ids = np.flatnonzero((source.geom_bodyid == source.body(body).id) & (source.geom_contype != 0))
    for g in ids:
        geom = source.geom(g)
        other = model.geom(geom.name)
        for key in ("type", "size", "pos", "quat", "friction", "priority", "solimp", "solref"):
            np.testing.assert_allclose(getattr(other, key), getattr(geom, key))


def run(scene, task, optimizer_seed, count, seconds, goal, out, parity=False):
    src = mujoco.MjModel.from_binary_path(str(scene))
    name = "spot_box_push" if task == "box" else "spot_cone_upright"
    body = "box_body" if task == "box" else "traffic_cone"
    configure = SpotBoxPush._process_spec

    def process(self):
        configure(self)
        shared_box(self, src)

    SpotBoxPush._process_spec = process
    if count:
        set_config_overrides(name, ControllerConfig, {"control_freq": 10.})
        set_config_overrides(name, CrossEntropyMethodConfig, {"num_rollouts": count})
    try:
        sim = get_simulation_backend("mujoco_hierarchical")(init_task=name)
        ctrl = make_controller(name, "cem")
    finally:
        SpotBoxPush._process_spec = configure
    for fixture in (sim.task.model, ctrl.rollout_backend.model):
        verify(src, fixture, body)
    m, data = sim.task.model, sim.task.data
    obj = m.body(body)
    mass = obj.mass.item()
    hook = ctypes.CDLL(None)
    hook.observe_setup(obj.id, m.body("arm_link_fngr").id, m.body("arm_link_wr1").id, m.geom("ground").id)
    sample = np.zeros((2, 11))
    assert sim.task.physics_substeps == 2 and abs(sim.task.dt-.02) < 1e-12
    np.random.seed(0)
    sim.task.reset()
    spawn = src.key("spawn").qpos
    data.qpos[:2], data.qpos[3:7], data.qpos[-7:] = spawn[:2], spawn[3:7], spawn[-7:]
    mujoco.mj_forward(m, data)

    def step(cmd, observe=True):
        hook.observe_begin(int(observe))
        sim.step(cmd)
        assert hook.observe_end(ctypes.c_void_p(sample.ctypes.data)) == 2

    neutral = sim.task.optimizer_warm_start()
    if task == "box":
        # Retract within ReLIC soft limits to clear the near cuboid during neutral settling.
        neutral[[4, 5]] = [-1.65, 2.3]
        data.qpos[-14:-7] = neutral[3:10]
        mujoco.mj_forward(m, data)
    for _ in range(50):
        step(neutral)
        if sample[:, 9:].any():
            raise ValueError("INVALID_SPAWN: ReLIC touched object in warmup")
    initial = data.qpos.copy()
    if (abs(sample[:, 2].mean()-mass*9.81) >= .05*mass*9.81 or
            np.linalg.norm(data.qvel[-3:]) >= .05 or np.linalg.norm(data.qvel[-6:-3]) >= .02):
        raise ValueError("INVALID_SPAWN: ReLIC object did not settle")
    if seconds == 0:
        out.mkdir(parents=True,exist_ok=True)
        (out/"warm.json").write_text(json.dumps(initial.tolist()))
        return
    if task == "box":
        for t in (sim.task, ctrl.task):
            t.config.goal_position = np.r_[goal, initial[-5]]
    ctrl.reset()
    np.random.seed(optimizer_seed)
    dt, period = sim.task.dt, 1/ctrl.controller_cfg.control_freq
    next_plan = 1.
    poses, vel, loads, lat, work, actions = [], [], [], [], [], []
    status = "time_limit"
    for i in range(round(seconds/dt)):
        t = 1.+i*dt
        if t+1e-9 >= next_plan:
            hook.observe_begin(0)
            begin = perf_counter()
            ctrl.update_states(MujocoState(t, data.qpos.copy(), data.qvel.copy(), data.mocap_pos.copy(),
                                           data.mocap_quat.copy(), sim.task.get_sim_metadata()))
            ctrl.update_action()
            lat.append(perf_counter()-begin)
            steps = hook.observe_end(None)
            assert 0 < steps <= 2*ctrl.num_timesteps*ctrl.optimizer_cfg.num_rollouts and steps%2 == 0
            work.append(steps//2)
            next_plan += period
        cmd = ctrl.action(t)
        actions.append(cmd.copy())
        step(cmd)
        poses.append(data.qpos.copy())
        vel.append(data.qvel.copy())
        loads.append(sample.copy())
        if (i+1)%5:
            continue
        q, v, f = np.asarray(poses), np.asarray(vel), np.asarray(loads)
        result = score(task, q, v, f[:,:,9].any(1), f[:,:,2].mean(1), f[:,:,10].sum(1),
                       mass, initial, goal)
        if (tilt(q[-5:]) > 60).any():
            status = "unstable"
            break
        if result["strict"]:
            status = "success"
            break
    ms = np.array(lat)*1000
    force = np.linalg.norm(f[:,:,3:6], axis=2)
    result.update(status=status, initial=initial.tolist(), sim_s=len(q)*dt,
                  contact_s=float((force>1).sum()*dt/2),
                  hand_impulse_Ns=float(np.linalg.norm(f[:,:,3:6].mean(1),axis=1).sum()*dt),
                  hand_tick_peak_N=float(np.linalg.norm(f[:,:,3:6].mean(1),axis=1).max()),
                  hand_peak_N=float(force.max()), p50_ms=np.median(ms), p95_ms=np.percentile(ms,95),
                  max_ms=ms.max(), calls=sum(work), scheduled=int(len(lat)*ctrl.num_timesteps*ctrl.optimizer_cfg.num_rollouts),
                  updates=len(lat), period_ms=period*1000, period_misses=int(sum(ms>period*1000)),
                  misses_100ms=int(sum(ms>100)), candidates=ctrl.optimizer_cfg.num_rollouts)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out/"episode.npz", qpos=q, qvel=v, substep_forces=f, action=actions,
                        planning_s=lat, calls=work, initial=initial)
    (out/"result.json").write_text(json.dumps(result, indent=2))
    if parity:
        np.random.seed(0)
        sim.task.reset()
        sim.reset_policy_state()
        data.qpos[:2], data.qpos[3:7], data.qpos[-7:] = spawn[:2], spawn[3:7], spawn[-7:]
        mujoco.mj_forward(m,data)
        if task == "box":
            data.qpos[-14:-7] = neutral[3:10]
            mujoco.mj_forward(m, data)
        for _ in range(50):
            step(neutral, False)
        for j, cmd in enumerate(actions):
            step(cmd, False)
            np.testing.assert_array_equal(data.qpos, q[j])
            np.testing.assert_array_equal(data.qvel, v[j])
    print(out, status, result["final_error"], sum(work), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", choices=["box", "cone"])
    p.add_argument("scene", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--seconds", type=float, required=True)
    p.add_argument("--goal", type=float, nargs=2)
    p.add_argument("--count", type=int, default=0)
    p.add_argument("--parity", action="store_true")
    a = p.parse_args()
    with TemporaryDirectory() as tmp:
        (Path(tmp)/"meshes").symlink_to(base.MODEL_PATH/"meshes", target_is_directory=True)
        base.MODEL_PATH = Path(tmp)
        run(a.scene, a.task, a.seed, a.count, a.seconds, np.asarray(a.goal), a.out, a.parity)
