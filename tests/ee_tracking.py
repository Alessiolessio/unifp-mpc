"""Measure three EE paths at their actual 50 Hz issue/execution times."""

import argparse
import json
from pathlib import Path

import numpy as np

from box_mpc import DT, WARMUP_TICKS, world_ee_target


def measure(folder):
    e = np.load(folder / "episode.npz")
    s = json.loads((folder / "summary.json").read_text())
    cone = s.get("task") == "cone"
    raw = (
        e["requests"][..., :3]
        if cone
        else np.load(folder / "plans.npz")["requested_prefix_m"]
    ).reshape(-1, 3)
    pre = e["qpos"][WARMUP_TICKS - 1 : -1]
    sph = e["policy_command"][WARMUP_TICKS:, 3:6]
    target = world_ee_target(pre[:, :7], sph)
    finger = e["ee_world"][WARMUP_TICKS - 1 :]

    def norm(v):
        return np.linalg.norm(v, axis=-1)

    velocity = np.diff(finger, axis=0) / DT
    speed = norm(velocity)
    acc = norm(np.diff(velocity, axis=0) / DT)
    raw_speed = norm(np.diff(raw, axis=0)) / DT
    effective_speed = norm(np.diff(target, axis=0)) / DT
    projection = norm(raw - target)
    tracking = norm(target - finger[1:])
    boundary = np.arange(4, len(raw_speed), 5)
    hit = e["contact_duty"][WARMUP_TICKS:] > 0
    force = norm(e["gripper_on_" + ("cone" if cone else "box") + "_N"][WARMUP_TICKS:])
    i = np.argmax(speed)

    def summary(x):
        return np.round(np.percentile(x, [50, 95, 100]), 4).tolist()

    return dict(
        raw_speed=summary(raw_speed),
        projected_speed=summary(effective_speed),
        finger_speed=summary(speed),
        acceleration=summary(acc),
        raw_gap=summary(norm(raw - finger[1:])),
        clipping_gap=summary(projection),
        tracking_gap=summary(tracking),
        saturation_pct=100 * np.mean(projection > 0.00001),
        boundary_raw_max=float(raw_speed[boundary].max()),
        boundary_projected_p95_max=summary(effective_speed[boundary])[1:],
        finger_boundary=summary(speed[::5]),
        finger_interior=summary(speed[np.arange(len(speed)) % 5 != 0]),
        replan_raw_gap=summary(norm(raw[::5] - finger[:-1:5])),
        replan_effective_gap=summary(norm(target[::5] - finger[:-1:5])),
        first_preexisting_raw_error=norm(raw[0] - finger[0]),
        tracking_loaded=summary(tracking[hit]) if hit.any() else None,
        tracking_unloaded=summary(tracking[~hit]) if not hit.all() else None,
        peak_finger_tick=dict(
            control_s=(i + 1) * DT,
            force_N=force[i],
            object_linear_speed=norm(e["qvel"][WARMUP_TICKS + i, -6:-3]),
            object_angular_speed=norm(e["qvel"][WARMUP_TICKS + i, -3:]),
        ),
        polar_flip_ticks=int(np.sum(abs(np.diff(sph[:, 2])) > np.pi / 2)),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("episodes", type=Path, nargs="+")
    print(json.dumps({str(f): measure(f) for f in p.parse_args().episodes}, indent=2))
