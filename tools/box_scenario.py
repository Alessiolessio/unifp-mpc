"""Generate one reproducible, matched-physics Spot/box scene and relative XY goal."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from prepare_scene import InvalidSpawn, scene_template


# Forward gap, lateral half-width, box yaw and goal bearing half-angles (deg),
# push length, floor and hand sliding friction. Relative poses use the initial body heading.
TIERS = {
    "easy":   ((1.04, 1.10), .05, 15, 12, (.08, .22), (.24, .36), (.27, .43)),
    "medium": ((1.12, 1.34), .12, 35, 25, (.16, .38), (.20, .44), (.22, .50)),
    "hard":   ((1.38, 1.65), .20, 45, 40, (.24, .52), (.18, .48), (.20, .54)),
    "oblique": ((1.30, 1.50), .05, 35, 15, (.24, .42), (.20, .44), (.22, .50)),
}

# Radius (m), unsigned approach bearing from body heading, unsigned push angle
# from approach (degrees), displacement (m). Each angle gets an independent sign.
POLAR_TIERS = {
    "extended": ((1.75, 2.45), (0, 25), (0, 20), (.25, .55)),
    "crossing": ((1.35, 1.90), (30, 70), (25, 65), (.25, .55)),
    "general": ((1.45, 2.30), (55, 110), (45, 110), (.30, .70)),
}
DIFFICULTIES = (*TIERS, *POLAR_TIERS)


def episode_seconds(difficulty):
    return {"extended": 15., "crossing": 15., "general": 20.}.get(difficulty, 9.)


def generate(folder, seed, long_push=False, ground_mu=None, hand_mu=None, goal_body=None,
             *, difficulty=None, model=None):
    rng = np.random.default_rng(seed)
    if difficulty in POLAR_TIERS:
        radius, approach, push_angle, lengths = POLAR_TIERS[difficulty]
        floor_range, hand_range = (.18, .48), (.20, .54)
        robot_xy = rng.uniform(-.4, .4, 2)
        yaw, box_angle = np.deg2rad(rng.uniform([-180, -90], [180, 90]))
        az = np.deg2rad(rng.uniform(*approach) * rng.choice([-1, 1]))
        box_offset = rng.uniform(*radius) * np.array([np.cos(az), np.sin(az)])
        bearing = yaw + az + np.deg2rad(rng.uniform(*push_angle) * rng.choice([-1, 1]))
    elif difficulty:
        gap, side, angle, spread, lengths, floor_range, hand_range = TIERS[difficulty]
        robot_xy = rng.uniform(-.4, .4, 2)
        yaw, box_angle = np.deg2rad(rng.uniform([-180, -angle], [180, angle]))
    else:
        gap, side, spread = (1.035, 1.105), .085, 30
        lengths = (.48, .70) if long_push else (.12, .38)
        floor_range, hand_range = (.20, .46), (.22, .50)
        robot_xy = rng.uniform(-.12, .12, 2)
        yaw, box_angle = np.deg2rad(rng.uniform([-15, -20], [15, 20]))
    rotate = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    if difficulty not in POLAR_TIERS:
        box_offset = rng.uniform([gap[0], -side], [gap[1], side])
        bearing = yaw + np.deg2rad(rng.uniform(-spread, spread))
    if difficulty == "oblique":
        # Rotate approach and push together to exercise turning before contact.
        obliquity = rng.choice([-1, 1]) * np.deg2rad(rng.uniform(30, 50))
        c, s = np.cos(obliquity), np.sin(obliquity)
        box_offset = np.array([[c, -s], [s, c]]) @ box_offset
        bearing += obliquity
    box_xy = robot_xy + rotate @ box_offset
    length = rng.uniform(*lengths)
    offset = length * np.array([np.cos(bearing), np.sin(bearing)])
    if goal_body is not None:
        if not np.isfinite(goal_body).all():
            raise ValueError("Goal displacement must be finite")
        offset = rotate @ goal_body
    sampled_floor, sampled_hand = rng.uniform(*floor_range), rng.uniform(*hand_range)
    floor = float(sampled_floor if ground_mu is None else ground_mu)
    hand = float(sampled_hand if hand_mu is None else hand_mu)
    if not all(np.isfinite(mu) and mu > 0 for mu in (floor, hand)):
        raise ValueError("Friction coefficients must be finite and positive")
    if model is None:
        model = scene_template()
    model.pair("box_ground").friction[:2] = floor
    model.geom("box_collision").friction[0] = hand
    q = model.key("spawn").qpos
    for joint, xy, angle in (("base", robot_xy, yaw), ("box_joint", box_xy, yaw + box_angle)):
        i = model.joint(joint).qposadr.item()
        q[i:i+2] = xy
        q[i+3:i+7] = [np.cos(angle/2), 0., 0., np.sin(angle/2)]
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("spawn").id)
    mujoco.mj_forward(model, data)
    box, ground = model.geom("box_collision").id, model.geom("ground").id
    if any(box in c.geom and ground not in c.geom for c in data.contact):
        raise InvalidSpawn("Robot and box are in contact at spawn")
    folder.mkdir(parents=True, exist_ok=True)
    mujoco.mj_saveModel(model, str(folder / "scene.mjb"))
    config = dict(scenario_seed=seed, long_push=long_push, robot_xy_m=robot_xy.tolist(),
                  robot_yaw_rad=yaw, box_xy_m=box_xy.tolist(), box_yaw_rad=yaw + box_angle,
                  goal_offset_m=offset.tolist(), ground_mu=floor, hand_mu=hand)
    if difficulty:
        config["difficulty"] = difficulty
    (folder / "scenario.json").write_text(json.dumps(config, indent=2) + "\n")
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    ranges = parser.add_mutually_exclusive_group()
    ranges.add_argument("--long-push", action="store_true")
    ranges.add_argument("--difficulty", choices=DIFFICULTIES)
    parser.add_argument("--ground-mu", type=float)
    parser.add_argument("--hand-mu", type=float)
    parser.add_argument("--goal-body", type=float, nargs=2, metavar=("DX", "DY"),
                        help="Goal displacement from the box, expressed in the initial robot yaw frame (m)")
    args = parser.parse_args()
    print(generate(args.out, args.seed, args.long_push, args.ground_mu, args.hand_mu,
                   args.goal_body, difficulty=args.difficulty))
