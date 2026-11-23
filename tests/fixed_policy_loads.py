"""Paired fixed-command trials. Run development first, then the frozen held-out protocol."""
import argparse
import csv
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np

from base_load_mpc import effective_velocity
from box_mpc import world_ee_target, yaw_of
from native_probe import DriveProbe
from prepare_scene import scene_template
from run_world import command_schedule, spherical
from tether_mpc import CASES, prepare
from unifp_native import World

OUT = Path("artifacts/fixed_policy_loads")
DT = .02
N = 150
MODES = ("A", "B", "C")


def length(x):
    return np.linalg.norm(x, axis=-1)


def rms(x):
    return np.sqrt(np.mean(np.square(x)))


def first(mask, hold=1):
    hits = np.flatnonzero(np.convolve(mask.astype(int), np.ones(hold, int), "valid") == hold)
    return (hits[0] + hold)*DT if len(hits) else np.nan


def command(xyz, vel=(0., 0.)):
    c = np.zeros(12, np.float32)
    c[:2] = vel
    c[3:6] = spherical(xyz)
    return c


def check_workspace(c):
    r, pitch, yaw = c[3:6]
    assert .15 < r < .55 and -40 < np.rad2deg(pitch) < 50 and abs(yaw) < np.pi/2


def ee_goal(pose, c):
    yaw = yaw_of(pose[:, 3:7])
    fx, fy, fz = c[6:9]
    force = np.column_stack((np.cos(yaw)*fx-np.sin(yaw)*fy,
                             np.sin(yaw)*fx+np.cos(yaw)*fy, np.full(len(yaw), fz)))
    return world_ee_target(pose[:, :7], np.broadcast_to(c[3:6], (len(pose), 3))) + force/200


def body_velocity(pose, vel):
    # Compare to root_lin_vel_b in the reward, including roll and pitch.
    w, v = pose[:, 3:4], -pose[:, 4:7]
    return (vel + 2*np.cross(v, np.cross(v, vel) + w*vel))[:, :2]


def payload_model(mass):
    m = mujoco.MjModel.from_binary_path("artifacts/spot.mjb")
    body = m.body("arm_link_fngr")
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("spawn").id)
    mujoco.mj_forward(m, d)
    unloaded_bias = d.qfrc_bias.copy()
    original_mass = m.body_subtreemass[0]
    # Rigid sphere of radius 4 cm centered at the existing finger COM. Its
    # isotropic inertia adds in any principal frame, with no parallel-axis term.
    body.mass[:] += mass
    body.inertia[:] += .4 * mass * .04**2
    mujoco.mj_setConst(m, d)
    mujoco.mj_resetDataKeyframe(m, d, m.key("spawn").id)
    mujoco.mj_forward(m, d)
    np.testing.assert_allclose(m.body_subtreemass[0]-original_mass, mass, atol=1e-12)
    jac = np.empty((3, m.nv))
    mujoco.mj_jacBodyCom(m, d, jac, None, body.id)
    np.testing.assert_allclose(d.qfrc_bias-unloaded_bias, -jac.T @ (mass*m.opt.gravity),
                               rtol=0, atol=1e-10)
    return m


def payload_measure(m, scene, warm, trace):
    # Replay the recorded ONNX actions from spawn through the same calibrated drive
    # for 200 Hz torques, checking the resulting joints at every policy tick.
    probe = DriveProbe(str(scene))
    motors = np.argsort(m.actuator_user[:, 0])
    joints = m.actuator_trnid[motors, 0]
    meta = m.actuator_user[motors]
    arm = np.array([m.joint(i).name.startswith("arm_") for i in joints])
    q0 = meta[:, 1].astype(np.float32)
    rows = []
    actual_q = []
    for action in np.concatenate((warm["action"], trace["action"])):
        target = q0 + np.float32(.25)*action
        for _ in range(4):
            q = probe.step(target)
            rows.append(probe.last_effort())
        actual_q.append(q)
    expected = np.concatenate((warm["qpos"], trace["qpos"]))[:, m.jnt_qposadr[joints]]
    np.testing.assert_array_equal(np.float32(actual_q), expected.astype(np.float32))
    tau = np.array(rows[-N*4:])[:, arm]
    limits = meta[arm, 4]
    d = mujoco.MjData(m)
    com = []
    for q in np.vstack((warm["qpos"][-1], trace["qpos"])):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        com.append(d.xipos[m.body("arm_link_fngr").id].copy())
    assert np.min(np.array(com)[:, 2]) > .04
    nonfinger = np.array([m.joint(j).name != "arm_f1x" for j in joints[arm]])
    return np.array(com), tau, limits, nonfinger


def prepare_case(task, case, dev, scene, box_mass):
    rng = np.random.default_rng(case)
    meta = {}
    if task == "box":
        m = mujoco.MjModel.from_binary_path("artifacts/spot_box.mjb")
        mass = (1., 2.)[case % 2] if dev else (1., 1., 2.)[case % 3]
        if box_mass is not None:
            mass = box_mass
        x = rng.uniform(1.015, 1.080)
        y = rng.uniform(-.045, .045)
        m.key("spawn").qpos[-7:-5] = [x, y]
        m.body("box_body").mass[:] *= mass
        m.body("box_body").inertia[:] *= mass
        mujoco.mj_setConst(m, mujoco.MjData(m))
        warm = command_schedule("push")[:50]
        xyz, vel = [.44, 0, -.17], [0, 0]
        meta.update(mass_kg=mass, box_x_m=x, box_y_m=y)
    elif task == "tether":
        name = "straight" if dev else ("diagonal", "heavy", "reverse_diagonal", "side")[case % 4]
        m = scene_template(tether=True)
        prepare(m, scene, name)
        speed = rng.uniform(.24, .32)
        bearing = np.deg2rad(rng.uniform(-12., 12.))
        vel = speed*np.array([np.cos(bearing), np.sin(bearing)])
        xyz = [.35, 0, .10]
        warm = command_schedule("stand")[:50]
        mass, mu, slack, yaw, _ = CASES[name]
        meta.update(mass_kg=mass, geometry=name, ground_mu=mu, spawn_slack_m=slack,
                    spawn_yaw_deg=yaw, walk_bearing_deg=np.rad2deg(bearing))
    else:
        mass = rng.uniform(.25, 1.0)
        xyz = [rng.uniform(.30, .42), rng.uniform(-.05, .05), rng.uniform(.03, .12)]
        vel = [.20 if task == "walking_payload" else 0., 0.]
        m = payload_model(mass)
        warm = np.tile(command(xyz, vel), (100, 1))
        meta.update(mass_kg=mass)
    if task != "tether":
        mujoco.mj_saveModel(m, str(scene))
    return m, warm, np.asarray(xyz), np.asarray(vel), meta


def run_case(task, case, dev, scene, folder, box_mass=None):
    m, warm_cmd, xyz, vel, meta = prepare_case(task, case, dev, scene, box_mass)
    w = World(str(scene), "artifacts/encoder_actor.onnx")
    warm = w.run(warm_cmd)
    snap = w.snapshot()
    q0 = warm["qpos"][-1]
    a = command(xyz, vel)
    if task in ("box", "tether"):
        assert not warm["contact_duty"].any() and not warm["other_robot_box_impulse_Ns"].any()
    if task == "tether":
        assert not warm["tension_N"].any() and not warm["gripper_touch"].any()
        m.tendon_limited[m.tendon("tow_cable").id] = False
        free_path = scene.with_name("untethered.mjb")
        mujoco.mj_saveModel(m, str(free_path))
        free = World(str(free_path), "artifacts/encoder_actor.onnx")
        free_warm = free.run(warm_cmd)
        np.testing.assert_array_equal(free_warm["qpos"], warm["qpos"])
        np.testing.assert_array_equal(free_warm["action"], warm["action"])
        twin = free.snapshot()
    payload = "payload" in task
    if payload:
        free = World("artifacts/spot.mjb", "artifacts/encoder_actor.onnx")
        free.run(warm_cmd)
        twin = free.snapshot()
        free_nominal = free.run(np.tile(a, (N, 1)))["ee_world"][:, 2]
    magnitude = {"box": 12., "tether": 16.}.get(task, 2.)
    rows = []
    for level in (-magnitude, magnitude):
        b, c = a.copy(), a.copy()
        if task == "tether":
            b[9:11] = level * vel/length(vel)
            c[:2] = vel+b[9:11]/200
            assert np.all(abs(c[:2]) <= [.6, .4])
            match = length(effective_velocity(b, [0, 0])-effective_velocity(c, [0, 0]))
        else:
            b[6 if task == "box" else 8] = level
            c[3:6] = spherical(xyz+b[6:9]/200)
            check_workspace(c)
            match = length(ee_goal(q0[None], b)-ee_goal(q0[None], c)).item()
        assert match < 1e-6
        for mode, cmd in zip(MODES, (a, b, c), strict=True):
            if mode == "A" and level > 0:
                continue
            check_workspace(cmd)
            w.restore(snap)
            u = np.tile(cmd, (N, 1))
            t = w.run(u)
            np.testing.assert_array_equal(t["policy_command"], u)
            pose, qvel = t["qpos"], t["qvel"]
            if task != "tether":
                match = length(ee_goal(pose, b)-ee_goal(pose, c)).max()
                assert match < 1e-6
            finger = t["ee_world"]
            goal = ee_goal(pose, cmd)
            v = body_velocity(pose, qvel[:, :3])
            tilt = np.rad2deg(np.arccos(np.clip(1-2*np.sum(pose[:, 4:6]**2, axis=1), -1., 1.)))
            row = dict(task=task, case=case, mode=mode, level_N=0 if mode == "A" else level,
                       **meta, vx=cmd[0], vy=cmd[1], nominal_x=xyz[0], nominal_y=xyz[1],
                       nominal_z=xyz[2], match_residual=match, max_tilt_deg=tilt.max(),
                       ee_rms_m=rms(length(finger-goal)))
            series = dict(pose=pose, finger=finger, action=t["action"], command=cmd,
                          start=q0, effective_ee=goal, body_velocity=v)
            if task in ("box", "tether"):
                xy = pose[:, -7:-5] - q0[-7:-5]
                direction = 0 if task == "box" else yaw_of(q0[3:7])+np.arctan2(vel[1], vel[0])
                along = np.array([np.cos(direction), np.sin(direction)])
                progress = xy @ along
                row.update(progress_m=progress[-1],
                           lateral_m=np.abs(xy @ [-along[1], along[0]]).max(),
                           peak_box_speed_m_s=length(qvel[:, -6:-4]).max(),
                           nongripper_Ns=t["other_robot_box_impulse_Ns"].sum())
                series.update(box_progress=progress, box_velocity=qvel[:, -6:-4])
            if task == "box":
                force = t["gripper_on_box_N"]
                row.update(impulse_x_Ns=DT*force[:, 0].sum(), mean_Fx_N=force[:, 0].mean(),
                           peak_Fx_N=t["peak_forward_N"].max(),
                           mean_hand_resultant_N=length(force).mean(),
                           contact_duty=t["contact_duty"].mean())
                series.update(gripper_N=force, contact_duty=t["contact_duty"])
            elif task == "tether":
                tension = t["tension_N"]
                taut = tension > .1
                transition = np.flatnonzero(taut.any(1))
                eff = effective_velocity(cmd, [0, 0])
                e = length(v-eff)
                free.restore(twin)
                unloaded = free.run(u)
                assert not unloaded["tension_N"].any()
                np.testing.assert_allclose(unloaded["qpos"][:, -7:-5], np.tile(q0[-7:-5], (N, 1)),
                                           rtol=0, atol=1e-10)
                free_v = body_velocity(unloaded["qpos"], unloaded["qvel"][:, :3])
                row.update(unloaded_v_rms=rms(length(free_v-eff)),
                           loaded_minus_free_speed_m_s=np.mean((v-free_v) @ (vel/length(vel))))
                n = transition[0] if len(transition) else N
                recovery = first(e[n:] <= .08, 10) if n <= N-10 else np.nan
                row.update(time_25cm_s=first(progress >= .25), taut_s=first(taut.any(1)),
                           post_taut_hold_s=recovery,
                           taut_window_v_rms=rms(e[n:n+25]) if n < N else np.nan,
                           nominal_v_rms=rms(length(v-vel)), effective_v_rms=rms(e),
                           tension_mean_N=tension.mean(), tension_peak_N=tension.max(),
                           taut_duty=taut.mean(), blocked_ticks=t["cable_blocked"].sum(),
                           hand_touch_ticks=t["gripper_touch"].sum())
                series.update(tension_N=tension, effective_v=eff, unloaded_v=free_v)
            else:
                free.restore(twin)
                unloaded = free.run(u)
                com, tau, limits, nonfinger = payload_measure(m, scene, warm, t)
                z = finger[:, 2]
                nominal = ee_goal(pose, a)[:, 2]
                clipped = abs(tau) >= limits-1e-5
                row.update(unloaded_posture_rms_m=rms(z-free_nominal),
                           nonfinger_clip_fraction=clipped[:, nonfinger].mean(),
                           height_m=z.mean(), nominal_z_rms_m=rms(z-nominal),
                           effective_z_rms_m=rms(z-goal[:, 2]),
                           unloaded_nominal_z_rms_m=rms(unloaded["ee_world"][:, 2]
                                                       - ee_goal(unloaded["qpos"], a)[:, 2]),
                           load_height_shift_m=np.mean(z-unloaded["ee_world"][:, 2]),
                           payload_dz_m=com[-1, 2]-com[0, 2],
                           payload_travel_m=length(com[-1]-com[0]),
                           arm_tau_peak_Nm=abs(tau).max(), arm_tau_rms_Nm=rms(tau),
                           arm_clip_fraction=clipped.mean(),
                           arm_peak_limit_fraction=np.max(abs(tau)/limits))
                series.update(payload_com=com[1:], start_com=com[0], arm_effort=tau,
                              unloaded_z=unloaded["ee_world"][:, 2], nominal_height=nominal,
                              free_nominal=free_nominal)
            np.savez_compressed(folder/f"{task}_{case}_{mode}{row['level_N']:+g}.npz", **series)
            rows.append(row)
        # First-tick actions have identical physical state, history and actuator queues.
        for mode, cmd in (("B", b), ("C", c)):
            w.restore(snap)
            tick = w.run(cmd[None, :])
            with np.load(folder/f"{task}_{case}_{mode}{level:+g}.npz") as saved:
                np.testing.assert_array_equal(saved["pose"][0], tick["qpos"][0])
                np.testing.assert_array_equal(saved["action"][0], tick["action"][0])
            if mode == "B":
                b_action = tick["action"][0]
            else:
                delta = abs(b_action-tick["action"][0]).max()
                for row in rows[-2:]:
                    row["first_action_BC_max"] = delta
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("split", choices=("development", "held"))
    args = p.parse_args()
    dev = args.split == "development"
    folder = OUT/args.split
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    with TemporaryDirectory(dir=OUT) as tmp:
        for task in ("box", "tether", "standing_payload", "walking_payload"):
            for case in (range(2) if dev else range(100, 112)):
                rows.extend(run_case(task, case, dev, Path(tmp)/"scene.mjb", folder))
            print(task, len(rows), flush=True)
        if not dev:
            # Every heavier push also gets its identical placement with a 1 kg box.
            twins = folder/"one_kg_twins"
            twins.mkdir(exist_ok=True)
            for case in range(101, 112, 3):
                rows.extend(run_case("box", case, dev, Path(tmp)/"scene.mjb", twins, box_mass=1.))
    with (folder/"metrics.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in rows for k in row)),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
