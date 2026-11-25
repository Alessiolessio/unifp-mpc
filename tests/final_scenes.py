"""Materialize final scenes before optimization; publish exact physical manifests."""
import csv
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from box_scenario import episode_seconds, generate
from cone_mpc import cone_model, generate_cone
from prepare_scene import InvalidSpawn, scene_template
from run_world import command_schedule
from tether_mpc import CASES, prepare
from unifp_native import World

ROOT = Path("artifacts/final")
MANIFEST = Path("docs/final_scenes.csv")
POLICY = "artifacts/encoder_actor.onnx"


def scenes():
    for task, seed, tiers in [
        ("box", 3001, sum(([k]*3 for k in ("easy","medium","extended","crossing","general")), [])),
        ("cone", 3101, ["near"]*4+["varied"]*4+["general"]*4),
        ("tether", 3201, ["low"]*4+["mid"]*4+["steep"]*4),
        ("box", 3301, ["easy"]*6), ("cone", 3401, ["general"]*6)
    ]:
        away_challenge = seed == 3401
        for i, tier in enumerate(tiers):
            if away_challenge:
                while True:
                    rng = np.random.default_rng(seed+i)
                    rng.uniform(1.22,1.60)
                    rng.uniform(-75,75)
                    rng.uniform(-180,180)
                    if min((a := rng.uniform(0,360)), 360-a) <= 30:
                        break
                    seed += 1
            yield ("challenge" if seed>=3301 else "general"), task, seed+i, tier


def build(block, task, seed, tier, folder):
    kg = 3. if block=="challenge" else 1.
    ratio = (5/3 if task=="box" else 1.25) if block=="challenge" else 1.
    if task=="box":
        s = generate(folder, seed, difficulty=tier, box_mass=kg)
        seconds = episode_seconds(tier)
    elif task=="cone":
        s = generate_cone(folder, seed, cone_model(), tier)
        kg, seconds = 3.36, 12.
        if block=="challenge":
            diff = s["relative_axis_bearing_deg"]-s["plate_bearing_deg"]
            assert abs((diff+180)%360-180) <= 30
    else:
        rng = np.random.default_rng(seed)
        lo, hi = {"low":(0,15), "mid":(15,35), "steep":(35,55)}[tier]
        kg, mu, slack, yaw, bearing = [rng.uniform(a,b) for a,b in
                                        [(1,3),(.27,.40),(.04,.14),(-18,18),(lo,hi)]]
        bearing *= rng.choice([-1,1])
        case = str(seed)
        CASES[case] = kg, mu, slack, yaw, bearing
        delta, limit = prepare(scene_template(tether=True), folder/"scene.mjb", case)
        s = dict(box_mass_kg=kg, ground_mu=mu, slack_m=slack, yaw_deg=yaw,
                 bearing_deg=bearing, goal_offset_m=delta.tolist(), limit_m=limit)
        seconds = 9.
    scene = folder/"scene.mjb"
    live = World(str(scene), POLICY)
    if ratio != 1:
        from unifp_native import Rollouts
        pool = Rollouts(live, 1)
        live = pool.execution_world(ratio)
        live.save_model(str(folder/"execution.mjb"))
    routine = "stand" if task=="tether" else "push"
    warm = live.run(command_schedule(routine)[:50])
    q = warm["qpos"][-1]
    if warm["contact_duty"].any() or warm[f"other_robot_{'cone' if task=='cone' else 'box'}_impulse_Ns"].any():
        raise InvalidSpawn("Warmup touched robot")
    if task=="cone" and (warm["gripper_touch"].any() or
            not 70 < np.degrees(np.arccos(1-2*np.sum(q[-3:-1]**2))) < 130 or
            np.linalg.norm(warm["qvel"][-1,-6:-3])>.02 or np.linalg.norm(warm["qvel"][-1,-3:])>.05
            or warm["ground_on_cone_N"][-1,2]<.5*kg*ratio*9.81):
        raise InvalidSpawn("Cone did not settle")
    if task=="tether" and (warm["tension_N"].any() or warm["gripper_touch"].any()):
        raise InvalidSpawn("Cable or hand loaded in warmup")
    (folder/"scenario.json").write_text(json.dumps(s, indent=2))
    relic_root = relic_object = None
    if task != "tether" and block == "general":
        import subprocess
        from final_pilots import relic
        try:
            relic(task, scene, folder, 0, 0., 6 if task=="box" else 8, capture=True)
        except subprocess.CalledProcessError as e:
            if "INVALID_SPAWN:" in e.stderr:
                raise InvalidSpawn(e.stderr.split("INVALID_SPAWN:")[-1].strip()) from e
            raise RuntimeError(e.stderr) from e
        other = json.loads((folder/"warm.json").read_text())
        relic_root, relic_object = other[:7], other[-7:]
    model = mujoco.MjModel.from_binary_path(str(scene))
    plant = folder/"execution.mjb" if ratio != 1 else scene
    return dict(model_sha256=hashlib.sha256(scene.read_bytes()).hexdigest(),
                plant_sha256=hashlib.sha256(plant.read_bytes()).hexdigest(), block=block, task=task, scene_seed=seed, tier=tier, seconds=seconds,
                nominal_kg=kg, execution_kg=kg*ratio, mass_scale=ratio,
                spawn_qpos=json.dumps(model.key("spawn").qpos.tolist()), scenario=json.dumps(s),
                relic_root=json.dumps(relic_root), relic_object=json.dumps(relic_object),
                settled_root=json.dumps(q[:7].tolist()), settled_object=json.dumps(q[-7:].tolist()),
                goal_xy=json.dumps((q[-7:-5]+s["goal_offset_m"]).tolist()) if task!="cone" else "")


if __name__ == "__main__":
    rows, exclusions = [], []
    for block, task, seed, tier in scenes():
        original = seed
        while True:
            try:
                row = build(block, task, seed, tier, ROOT/str(seed))
                break
            except InvalidSpawn as e:
                exclusions.append(dict(task=task, seed=seed, tier=tier, reason=str(e)))
                seed += 10000
        rows.append(row)
    with MANIFEST.open("w") as f:
        writer = csv.DictWriter(f, rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (ROOT/"exclusions.json").write_text(json.dumps(exclusions))
