"""Compare contact and cost ranks on the same forecast batch at archived slip states."""

import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from box_mpc import BoxPlanner, box_cost, warmup_target
from cem import DT, PREFIX
from recover_box import CASES
from run_world import command_schedule
from unifp_native import Rollouts, World


def surface(model, poses, duty, goal):
    """Reconstruct tick-end geometry. The separate impulse metric uses all physics substeps."""
    data = mujoco.MjData(model)
    box = model.geom("box_collision").id
    hand = {model.body(name).id for name in ("arm_link_fngr", "arm_link_wr1")}
    points, alignment = [], []
    for k in np.flatnonzero(duty):
        data.qpos[:] = poses[k]
        mujoco.mj_forward(model, data)
        center = data.geom_xpos[box]
        direction = goal - center[:2]
        direction /= max(np.linalg.norm(direction), 1e-12)
        for contact in data.contact:
            a, b = contact.geom
            if box not in (a, b) or model.geom_bodyid[b if a == box else a] not in hand:
                continue
            points.append((contact.pos - center) @ data.geom_xmat[box].reshape(3, 3))
            normal = contact.frame[:2] * (1 if b == box else -1)
            alignment.append(normal @ direction)
    if points:
        return dict(points=len(points), box_local_xyz_m=np.mean(points, axis=0).round(3).tolist(),
                    mean_normal_dot_goal=round(float(np.mean(alignment)), 3))


def inspect(case, archive, checkpoints):
    folder = Path(archive)
    summary = json.loads((folder / "summary.json").read_text())
    model = mujoco.MjModel.from_binary_path(summary["scene"])
    live = World(summary["scene"], "artifacts/encoder_actor.onnx")
    pool = Rollouts(live, 8)
    warm = live.run(command_schedule("push")[:50])
    q = warm["qpos"][-1]
    begin = q[-7:-5].copy()
    goal = np.array(summary["goal_xy_m"])
    length = np.linalg.norm(goal - begin)
    axis = (goal - begin) / length
    batch = {}

    def evaluate(snapshot, commands):
        trace = pool.evaluate(snapshot, commands)
        if i in checkpoints:
            batch.update(trace)
        return trace

    cfg = summary["scenario"]
    planner = BoxPlanner(live, q[-7:-4], warmup_target(warm), summary["candidates"], 1, 8,
                         summary["seed"], mobile=True, objective="bounded",
                         pool=SimpleNamespace(evaluate=evaluate),
                         base_control=summary.get("base_control"), horizon=summary.get("horizon_s", .5),
                         workspace_yaw=cfg["robot_yaw_rad"] if "difficulty" in cfg else None)
    with np.load(folder / "episode.npz") as episode:
        archived_qpos = episode["qpos"]
    for i in range(max(checkpoints) + 1):
        np.testing.assert_array_equal(q, archived_qpos[49 + i * PREFIX])
        xy = q[-7:-5]
        prefix, predicted, _ = planner.plan(live, xy, goal, root=q[:7])
        if i in checkpoints:
            boxes = batch["qpos"][:, :, -7:-5]
            local = begin + np.clip((xy - begin) @ axis + .08, 0, length) * axis
            running = np.sum((local - boxes)**2, axis=2)
            ranks = []
            for proximity in (0., 1.):
                costs = box_cost(batch, xy, local, mobile=True, bounded=True, proximity=proximity)
                costs += .5 * (running.mean(1) - running[:, -1])
                ranks.append(int(np.argmin(costs)))
                if proximity == 0.:
                    finite = costs[np.isfinite(costs)]
                    spread = np.ptp(finite)
            selected = ranks[0]
            np.testing.assert_array_equal(batch["action"][selected, :PREFIX], predicted["action"])
            gain = 1000 * (np.linalg.norm(goal - xy) - np.linalg.norm(goal - boxes[:, -1], axis=1))
            ranks.append(int(np.argmax(np.where(np.isfinite(costs), gain, -np.inf))))
            contacts = (batch["contact_duty"] > 0).any(1)
            direction = (goal - xy) / np.linalg.norm(goal - xy)
            impulse = batch["gripper_on_box_N"][:, :, :2].sum(1) * DT
            hit = surface(model, batch["qpos"][selected], batch["contact_duty"][selected], goal)
            print(json.dumps(dict(case=case, t=i/10, candidates_contact=int(contacts.sum()),
                                  directional_cost_span=float(spread),
                                  index_dir_prox_oracle=ranks, goal_gain_mm=gain[ranks].round(2).tolist(),
                                  signed_impulse_Ns=(impulse @ direction)[ranks].round(3).tolist(),
                                  predicted_surface=hit)))
        q = live.run(prefix)["qpos"][-1]

    with np.load(folder / "episode.npz") as episode:
        for a, b in zip(checkpoints, checkpoints[1:]):
            cut = slice(50 + PREFIX*a, 50 + PREFIX*b)
            hit = surface(model, episode["qpos"][cut], episode["contact_duty"][cut], goal)
            print(json.dumps(dict(case=case, executed_window=[a/10, b/10], surface=hit)))


if __name__ == "__main__":
    for name, (archive, _) in CASES.items():
        inspect(name, archive, [40, 55, 65, 80])
