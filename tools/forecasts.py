"""Capture just the winning rollout, before the batch buffers go out of scope."""
import numpy as np

from cem import PREFIX


def winner(batch, index, commands, pre_root):
    keys = ("qpos", "qvel", "policy_command", "ee_world", "contact_duty", "gripper_on_box_N",
            "gripper_on_cone_N", "gripper_on_wall_N", "ground_on_cone_N", "wall_normal_N",
            "tension_N", "eye_to_hitch_m")
    row = {k: batch[k][index].copy() for k in keys if k in batch}
    row.update(request=commands[index].copy(), pre_root=pre_root.copy())
    return row


def pack(rows, start_tick):
    if not rows:
        return {}
    return {"forecast_" + k: np.stack([r[k] for r in rows]) for k in rows[0]} | {
        "forecast_start_tick": start_tick + PREFIX*np.arange(len(rows))}
