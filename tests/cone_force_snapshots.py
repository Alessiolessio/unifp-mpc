"""Signed yaw-frame force interventions on archived, exactly restored cone contact states."""
import csv
from pathlib import Path

import numpy as np

from box_mpc import world_ee_target
from cone_contact_wrench import analyze
from cone_mpc import angle_deg, base_tilt, cone_axis
from measure_force import world_force
from run_world import command_schedule
from unifp_native import World

ROOT = Path("artifacts/ee_study")
OUT = Path("artifacts/cone_force")
POLICY = "artifacts/encoder_actor.onnx"
# First scored tick in the original full episode, including its 50 warmup ticks.
SNAPS = (("approach", "near-upright_buffer/0", 50),
         ("shaft", "near-upright_buffer/0", 150),
         ("partial", "near-upright_buffer/0", 280),
         ("near_upright", "near-upright_buffer/0", 365),
         ("away_plate", "general-upright_buffer/82", 507),
         ("away_shaft", "general-plate_access--joint-proposals/82", 445))


def probe(label, path, start):
    folder = ROOT/path
    with np.load(folder/"episode.npz") as f:
        archive = {k: f[k] for k in ("action", "qpos", "requests", "gripper_on_cone_N")}
    raw = archive["requests"].reshape(-1, 6)
    live = World(str(folder/"scene.mjb"), POLICY)
    live.run(command_schedule("push")[:50])
    if start > 50:
        live.run(raw[:start-50])
    state = live.snapshot()
    q0 = archive["qpos"][start-1]
    u = np.zeros((25, 9), np.float32)
    u[:, :6] = raw[start-50:start-25]
    rows = []
    for axis, level in [(0, 0)] + [(i,v) for i in range(3) for v in (-12,12)]:
        u[:, 6:] = 0
        u[:, 6+axis] = level
        for mode in (("B", "C") if level else ("A",)):
            live.restore(state)
            t = live.run(u, matched_position=mode == "C")
            if not level:
                np.testing.assert_array_equal(t["qpos"], archive["qpos"][start:start+25])
            # Continue the archived zero-force winners open loop. This checks downstream
            # release under one common continuation without granting a new optimizer.
            tail = live.run(raw[start-25:])
            hold = ((angle_deg(tail["qpos"][:,-4:]) <= 10)
                    & (np.linalg.norm(tail["qvel"][:,-3:],axis=1) <= .25)
                    & (tail["ground_on_cone_N"][:,2] >= .5*3.36*9.81)
                    & ~tail["gripper_touch"] & (base_tilt(tail["qpos"]) < 30)
                    & (tail["qpos"][:,2] >= .35))
            released = len(hold) >= 25 and np.convolve(hold, np.ones(25), "valid").max() >= 25
            residual = 0.
            action_delta_bc = None
            if mode == "B":
                b_action = t["action"][0]
                # C on the exact same pre-action pose/history at every B tick, before branching.
                live.restore(state)
                for i, request in enumerate(u):
                    root = live.base_pose.copy()
                    snap = live.snapshot()
                    c = live.run(request[None], matched_position=True)["policy_command"][0]
                    live.restore(snap)
                    b = live.run(request[None])["policy_command"][0]
                    pb = world_ee_target(root, b[3:6]) + world_force(root,b)/200
                    pc = world_ee_target(root, c[3:6])
                    residual = max(residual, np.linalg.norm(pb-pc))
                assert residual < 1e-6
            elif mode == "C":
                assert not t["policy_command"][:, 6:].any()
                action_delta_bc = np.max(abs(b_action-t["action"][0]))
            replay = {k: np.concatenate((archive[k][:start], t[k])) for k in
                      ("action", "qpos", "gripper_on_cone_N")}
            w = analyze(folder, trace=replay, start=start)
            q = t["qpos"]
            a = cone_axis(q[:, -4:])
            a0 = cone_axis(q0[-4:])
            tip = angle_deg(q0[-4:]) - angle_deg(q[-1,-4:])
            com = q[:, -7:-4] + .24*a
            force = np.linalg.norm(t["gripper_on_cone_N"], axis=1)
            rows.append(dict(snapshot=label, tick=start, axis="xyz"[axis], level_N=level, mode=mode,
                             tilt_start_deg=angle_deg(q0[-4:]), tilt_gain_deg=tip,
                             alignment_gain=a[-1,2]-a0[2],
                             final_tip_rate_deg_s=(angle_deg(q[-2,-4:])-angle_deg(q[-1,-4:]))/.02,
                             com_travel_xy_m=np.linalg.norm((com[-1]-q0[-7:-4]-.24*a0)[:2]),
                             max_tilt_robot_deg=base_tilt(q).max(), peak_tick_N=force.max(),
                             first_BC_action_delta=action_delta_bc,
                             first_action_delta=np.max(abs(t["action"][0]-archive["action"][start])),
                             actual_virtual_N=np.linalg.norm(t["policy_command"][:,6:9],axis=1).mean(),
                             match_residual_m=residual, axial_m=w["impulse_weighted_axial_m"],
                             hand_impulse_Ns=sum(w["geom_Ns"].values()),
                             continuation_release=bool(released), continuation_final_deg=angle_deg(tail["qpos"][-1,-4:]),
                             continuation_other_Ns=tail["other_robot_cone_impulse_Ns"].sum(),
                             support_moment_Nms=w["supported_tipping_Nms"],
                             support_hand_Ns=w["supported_hand_Ns"], geoms=str(w["geom_Ns"])))
    return rows


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    with (OUT/"snapshots.csv").open("w") as f:
        for label, path, start in SNAPS:
            rows = probe(label,path,start)
            writer = csv.DictWriter(f, fieldnames=rows[0], lineterminator="\n")
            if f.tell() == 0:
                writer.writeheader()
            writer.writerows(rows)
            f.flush()
            print(label, flush=True)
