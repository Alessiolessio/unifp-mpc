"""Position-only UniFP MPC for uprighting SUMO's free traffic cone."""

from functools import partial
import json
from pathlib import Path
from time import perf_counter

import mujoco
import numpy as np

from box_mpc import BoxPlanner, WARMUP_TICKS, latency, warmup_target, yaw_of
from cem import DT, PREFIX
from prepare_scene import InvalidSpawn, scene_template
from run_world import command_schedule


CONE_XML = Path("../sumo/sumo/models/xml/objects/traffic_cone/traffic_cone.xml")
COM_HEIGHT = .24  # SUMO inertial origin along cone-local +Z (m)
SUMO_UPRIGHT = .9
CONE_TIERS = ("near", "varied", "general")


def cone_model():
    return scene_template(cone_xml=CONE_XML.resolve())


def show_cone_mesh(model):
    obj = model.body("traffic_cone").id
    model.geom_group[(model.geom_bodyid == obj) & (model.geom_contype != 0)] = 3
    model.site_rgba[model.site_bodyid == obj, 3] = 0.


def cone_axis(quat):
    """World direction of body +Z from a MuJoCo unit quaternion (wxyz)."""
    w, x, y, z = np.moveaxis(quat, -1, 0)
    return np.stack((2*(x*z + w*y), 2*(y*z - w*x), 1-2*(x*x + y*y)), axis=-1)


def angle_deg(quat):
    return np.rad2deg(np.arccos(np.clip(cone_axis(quat)[..., 2], -1., 1.)))


def base_tilt(qpos):
    w, x, y, z = qpos[:, 3:7].T
    return np.rad2deg(np.maximum(abs(np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y))),
                                 abs(np.arcsin(np.clip(2*(w*y-z*x), -1, 1)))))


def generate_cone(folder, seed, model, difficulty="near"):
    rng = np.random.default_rng(seed)
    distance, side, heading, azimuth, tilt = 1.15, 0., 0., 180., 90.
    if difficulty == "near" and seed:
        distance, side, heading, azimuth, tilt = rng.uniform(
            [1.10, -.10, -15., 155., 83.], [1.30, .10, 15., 205., 100.])
    elif difficulty in ("varied", "general"):
        varied = difficulty == "varied"
        radius = rng.uniform(1.18, 1.50) if varied else rng.uniform(1.22, 1.60)
        bearing = rng.uniform(-28., 28.) if varied else rng.uniform(-75., 75.)
        angle = np.deg2rad(bearing)
        distance, side = radius * np.cos(angle), radius * np.sin(angle)
        heading = rng.uniform(-180., 180.)
        azimuth = bearing + (rng.uniform(95., 265.) if varied else rng.uniform(0., 360.))
        tilt = 102.45  # passive side-rest angle for the SUMO plate and shaft
    elif difficulty != "near":
        raise ValueError(f"Unknown cone difficulty {difficulty}")
    yaw, az, tip = np.deg2rad([heading, heading + azimuth, tilt])
    q = model.key("spawn").qpos
    q[3:7] = [np.cos(yaw/2), 0., 0., np.sin(yaw/2)]
    q[-7:-4] = [distance*np.cos(yaw) - side*np.sin(yaw),
                distance*np.sin(yaw) + side*np.cos(yaw), .28]
    # Rz(az) Ry(tip) sets the axial bearing and tilt independently.
    c, s = np.cos(az/2), np.sin(az/2)
    q[-4:] = [c*np.cos(tip/2), -s*np.sin(tip/2), c*np.sin(tip/2), s*np.cos(tip/2)]
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("spawn").id)
    mujoco.mj_forward(model, data)
    obj = model.body("traffic_cone").id
    geoms = np.flatnonzero((model.geom_bodyid == obj) & (model.geom_contype != 0))
    floor = model.geom("ground").id

    def floor_gap():
        return min(mujoco.mj_geomDistance(model, data, floor, g, 2., None) for g in geoms)

    if difficulty != "near":
        q[-5] += .005 - floor_gap()
        mujoco.mj_resetDataKeyframe(model, data, model.key("spawn").id)
        mujoco.mj_forward(model, data)
    clearance = floor_gap()
    if clearance < 0 or any(obj in model.geom_bodyid[c.geom] for c in data.contact):
        raise InvalidSpawn("Cone overlaps another geom at reset")
    folder.mkdir(parents=True, exist_ok=True)
    mujoco.mj_saveModel(model, str(folder / "scene.mjb"))
    return dict(scenario_seed=seed, difficulty=difficulty, distance_m=distance, lateral_m=side,
                plate_range_m=np.hypot(distance, side),
                plate_bearing_deg=np.rad2deg(np.arctan2(side, distance)),
                robot_yaw_rad=yaw, relative_axis_bearing_deg=azimuth % 360, spawn_tilt_deg=tilt,
                spawn_floor_gap_m=clearance)


def cone_cost(trace, *, release=True):
    pose = trace["qpos"][..., -7:]
    axis = cone_axis(pose[..., 3:])
    center = pose[..., :3] + COM_HEIGHT * axis
    gap = np.linalg.norm(trace["ee_world"] - center, axis=-1)
    # Once upright, reward release. Continued COM proximity otherwise keeps the hand on the cone.
    approach = gap
    if release:
        approach = np.where(axis[..., 2] < SUMO_UPRIGHT, gap, .30 * trace["contact_duty"])
    cost = (1 - axis[..., 2]).mean(axis=1) + .005 * approach.mean(axis=1)
    return np.where(trace["valid"] & (trace["other_robot_cone_impulse_Ns"].max(axis=1) == 0),
                    cost, np.inf)


def clearance_cost(trace, *, running=False):
    gap = trace["chassis_cone_gap_m"]
    if running:
        penalty = .08 * (np.maximum(0., 1. - gap / .10)**2).mean(axis=1)
    else:
        # Approximately one foot radius at the closest predicted pass.
        penalty = .02 * np.maximum(0., 1. - gap.min(axis=1) / .04)**2
    cost = cone_cost(trace)
    return np.where(np.isfinite(cost), cost + penalty, np.inf)


def plate_access_cost(trace):
    """Add the extra finger-to-COM distance when the straight line crosses the cone plate."""
    pose = trace["qpos"][..., -7:]
    delta = trace["ee_world"] - pose[..., :3]
    v = pose[..., 4:]
    xyz = delta + 2*np.cross(v, np.cross(v, delta) - pose[..., 3:4]*delta)
    x, y, z = np.moveaxis(xyz, -1, 0)
    half, plate = .18415, .033
    distance = np.linalg.norm(xyz - [0, 0, COM_HEIGHT], axis=-1)
    # Intersect the finger-COM segment with the plate plane in object coordinates.
    scale = (COM_HEIGHT-plate) / np.maximum(COM_HEIGHT-z, COM_HEIGHT-plate)
    blocked = (z < plate) & (scale*np.maximum(abs(x), abs(y)) < half)
    # Unfold the shortest path around either nearer edge of the square.
    back = np.hypot(COM_HEIGHT-plate, half)
    via_x = np.hypot(np.hypot(abs(x)-half, z-plate) + back, y)
    via_y = np.hypot(np.hypot(abs(y)-half, z-plate) + back, x)
    extra = np.where(blocked & (cone_axis(pose[..., 3:])[..., 2] < SUMO_UPRIGHT),
                     np.minimum(via_x, via_y) - distance, 0.)
    cost = clearance_cost(trace)
    return np.where(np.isfinite(cost), cost + .005*extra.mean(axis=1), np.inf)


def cone_proposals(q, planner):
    """Sample a hand location and a compatible base pose, competing with ordinary CEM draws."""
    n = max(1, planner.count // 4)
    rng = planner.rng
    axis = cone_axis(q[-4:])
    target = q[-7:-4] + rng.uniform(0., .67, (n, 1))*axis + rng.normal(0., .12, (n, 3))
    yaw = yaw_of(q[3:7])
    turn = rng.uniform(-np.pi/2, np.pi/2, n)
    headings = np.column_stack((np.cos(yaw+turn), np.sin(yaw+turn)))
    desired_base = target[:, :2] - rng.uniform(.55, .85, (n, 1))*headings
    c, s = np.cos(yaw), np.sin(yaw)
    translation = (desired_base - q[:2]) @ np.array([[c, -s], [s, c]])
    target[:, :2] = target[:, :2] @ planner.rotation
    end = np.column_stack((target, translation/planner.nodes[-1], turn/planner.nodes[-1]))
    return ((end - planner.center) / planner.scale)[:, None, :]


def run_cone(args, folder, scenario, live, pool, display=None):
    start = perf_counter()
    warm = live.run(command_schedule("push")[:WARMUP_TICKS])
    q = warm["qpos"][-1]
    if (warm["gripper_touch"].any() or warm["other_robot_cone_impulse_Ns"].any()
            or not 70 < angle_deg(q[-4:]) < 130
            or np.linalg.norm(warm["qvel"][-1, -6:-3]) > .02
            or np.linalg.norm(warm["qvel"][-1, -3:]) > .05
            or warm["ground_on_cone_N"][-1, 2] < .5 * 3.36 * 9.81):
        raise InvalidSpawn("Expected a motionless, ground-supported fallen cone after warmup")
    anchor = warmup_target(warm)
    planner = BoxPlanner(live, q[-7:-4], anchor, args.candidates, args.iterations, args.workers,
                         args.seed, mobile=True, horizon=args.horizon, base_control="full",
                         workspace_yaw=scenario["robot_yaw_rad"], pool=pool,
                         sumo_sampling=args.sumo_sampling)
    score = {"upright_release": cone_cost, "upright_proximity": partial(cone_cost, release=False),
             "upright_clearance": partial(clearance_cost, running=True),
             "upright_buffer": clearance_cost, "plate_access": plate_access_cost}[args.objective]
    joint_proposals = getattr(args, "joint_proposals", False)
    traces = [warm]
    elapsed, valid, requests = [], [], []
    targets, projected, fingers = [], [], []
    held = 0
    status = "time_limit"
    if display:
        display.play(warm, None, None, None)
    for _ in range(int((args.seconds or 12.) / (PREFIX * DT) + 1e-9)):
        before = perf_counter()
        prefix, predicted, plan = planner.optimize(
            live, q[-7:-5], score, capture_path=display is not None or bool(args.out),
            propose=partial(cone_proposals, q) if joint_proposals else None)
        elapsed.append(perf_counter() - before)
        valid.append(int(plan["valid_candidates"]))
        if prefix is None:
            status = "no_valid_plan"
            break
        executed = live.run(prefix)
        if any(not np.array_equal(executed[k], v) for k, v in predicted.items()):
            raise RuntimeError("Cone prefix diverged from its native forecast")
        traces.append(executed)
        requests.append(prefix)
        if args.out:
            target, effective, finger = plan["view_paths"]
            targets.append(target)
            projected.append(effective)
            fingers.append(finger)
        q = executed["qpos"][-1]
        settled = ((angle_deg(executed["qpos"][:, -4:]) <= 10.)
                   & (np.linalg.norm(executed["qvel"][:, -3:], axis=1) <= .25)
                   & (executed["ground_on_cone_N"][:, 2] >= .5 * 3.36 * 9.81)
                   & ~executed["gripper_touch"]
                   & (base_tilt(executed["qpos"]) < 30) & (executed["qpos"][:, 2] >= .35))
        for tick in settled:
            held = held + 1 if tick else 0
        if display:
            display.play(executed, None, plan["view_paths"], elapsed[-1])
        if held >= 25:
            status = "success"
            break

    trace = {k: np.concatenate([t[k] for t in traces]) for k in warm}
    pose = trace["qpos"][:, -7:]
    angle = angle_deg(pose[:, 3:])
    axis = cone_axis(pose[:, 3:])
    first = np.flatnonzero(axis[WARMUP_TICKS:, 2] >= SUMO_UPRIGHT)
    ground = trace["ground_on_cone_N"][:, 2]
    forces = np.linalg.norm(trace["gripper_on_cone_N"], axis=1)
    com = pose[:, :3] + COM_HEIGHT * axis
    result = dict(task="cone", status=status, scenario=scenario, cem_seed=args.seed,
                  horizon_s=args.horizon, candidates=args.candidates, iterations=args.iterations,
                  base_control="full", objective=args.objective,
                  joint_proposals=joint_proposals,
                  workers=args.workers, sumo_sampling=args.sumo_sampling, control_limit_s=args.seconds or 12.,
                  start_angle_deg=angle[WARMUP_TICKS-1], final_angle_deg=angle[-1],
                  min_angle_deg=angle[WARMUP_TICKS-1:].min(), settled_s=held*DT,
                  sumo_first_s=(int(first[0])+1)*DT if len(first) else None,
                  final_angular_speed_rad_s=np.linalg.norm(trace["qvel"][-1, -3:]),
                  final_ground_z_N=ground[-1], final_gripper_N=forces[-1],
                  displacement_com_m=(com[-1]-com[WARMUP_TICKS-1]).tolist(),
                  peak_gripper_N=forces.max(), gripper_contact_ticks=int(np.sum(trace["contact_duty"] > 0)),
                  other_robot_cone_impulse_Ns=trace["other_robot_cone_impulse_Ns"].sum(),
                  min_chassis_gap_m=trace["chassis_cone_gap_m"][WARMUP_TICKS-1:].min(),
                  max_abs_roll_pitch_deg=base_tilt(trace["qpos"]).max(),
                  min_base_height_m=trace["qpos"][:, 2].min(),
                  control_s=(len(pose)-WARMUP_TICKS)*DT, mpc_updates=len(elapsed),
                  valid_candidates=sum(valid), planning=latency(elapsed), wall_s=perf_counter()-start)
    if args.out:
        result["scene"] = str(folder / "scene.mjb")
        (folder / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        np.savez_compressed(folder / "episode.npz", **trace, time_s=(np.arange(len(pose))+1)*DT,
                            requests=np.asarray(requests), targets=targets, projected=projected,
                            fingers=fingers,
                            planning_s=elapsed, valid_candidates=valid)
    return result
