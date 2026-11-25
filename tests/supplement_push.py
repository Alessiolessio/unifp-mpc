"""Replay the fixed 50 mm goal and test a development-calibrated ordinary reach."""
import csv
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np
from fixed_policy_loads import N, DT, command, ee_goal, length, rms, check_workspace, prepare_case
from final_policy import save
from physical_score import score
from unifp_native import World

ROOT = Path("artifacts/supplemental/push")
POLICY = "artifacts/encoder_actor.onnx"
GOAL = .05
OFFSETS = (.02, .035, .05)


def trial(case, offsets, cohort, scene, *, unobstructed=False, mass=1.):
    m, warm_cmd, xyz, vel, meta = prepare_case("box", case, False, scene, mass)
    if unobstructed:
        m.key("spawn").qpos[-7] += 3.
        meta["box_x_m"] = float(m.key("spawn").qpos[-7])
        mujoco.mj_saveModel(m, str(scene))
    world = World(str(scene), POLICY)
    warm = world.run(warm_cmd)
    initial = warm["qpos"][-1]
    assert not warm["contact_duty"].any()
    snap = world.snapshot()
    folder = ROOT/cohort/str(case)
    folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/"warm.npz", **warm)
    goal = initial[-7:-5] + [GOAL,0.]
    rows = []
    for arm, offset in offsets.items():
        world.restore(snap)
        cmd = command(xyz+[offset,0,0], vel)
        if arm == "B":
            cmd[6] = 12
        check_workspace(cmd)
        u = np.tile(cmd,(N,1))
        tr = world.run(u)
        np.testing.assert_array_equal(tr["policy_command"], u)
        q = tr["qpos"]
        nominal = cmd.copy()
        nominal[6:9] = 0
        pre = np.vstack((initial,q[:-1]))
        projected = ee_goal(pre,nominal)
        effective = ee_goal(pre,cmd)
        # Compare training-frame targets at the same post-action body pose as the measurement.
        tracking = tr["ee_world"] - ee_goal(q,cmd)
        if arm in ("B","C"):
            np.testing.assert_allclose(ee_goal(pre,command(xyz+[.06,0,0])),effective,atol=1e-7,rtol=0)
        force = tr["gripper_on_box_N"]
        loaded = tr["contact_duty"] > 0
        starts = loaded & ~np.r_[False,loaded[:-1]]
        along = q[:,-7]-initial[-7]
        obj_v = tr["qvel"][:,-6:-4]
        accel = np.diff(np.vstack((warm["qvel"][-1,-6:-4], obj_v)),axis=0)/DT
        stats = score("box",q,tr["qvel"],loaded,tr["ground_on_box_N"][:,2],
                      tr["other_robot_box_impulse_Ns"],mass,initial,goal)
        np.savez_compressed(folder/f"{arm}.npz", **tr, requested=u, nominal_pre=projected,
                            effective_pre=effective, effective_tracking_post=tracking, goal_xy=goal)
        active = np.flatnonzero(loaded)
        rows.append(dict(case=case,arm=arm,cohort=cohort,offset_m=offset, **meta, **stats,
                         end_settled=bool((length(q[-25:,-7:-5]-goal)<=.015).all() and
                                          (length(obj_v[-5:])<=.02).all()),
                         nominal_rms_mm=1000*rms(length(tr["ee_world"]-ee_goal(q,nominal))),
                         effective_rms_mm=1000*rms(length(tracking)),
                         effective_end_mm=1000*length(tracking[-1]),
                         finger_forward_m=tr["ee_world"][-1,0]-warm["ee_world"][-1,0],
                         base_dx_m=q[-1,0]-initial[0],base_dy_m=q[-1,1]-initial[1],
                         impulse_x_Ns=DT*force[:,0].sum(),impulse_y_Ns=DT*force[:,1].sum(),
                         abs_lateral_impulse_Ns=DT*abs(force[:,1]).sum(),
                         mean_Fx_N=force[:,0].mean(),mean_Fy_N=force[:,1].mean(),
                         peak_forward_N=tr["peak_forward_N"].max(), duty=tr["contact_duty"].mean(),
                         contact_s=DT*tr["contact_duty"].sum(), contact_bouts=int(starts.sum()),
                         first_contact_s=DT*(active[0]+1) if active.size else None,
                         last_contact_s=DT*(active[-1]+1) if active.size else None,
                         peak_speed=length(obj_v).max(),peak_tick_accel=length(accel).max(),
                         progress_m=along[-1], end_lateral_m=q[-1,-6]-initial[-6],
                         workspace_saturated_ticks=0))
    return rows


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=ROOT) as tmp:
        scene = Path(tmp)/"scene.mjb"
        abc = {"A":0., "B":0., "C":.06}
        archive = list(csv.DictReader(open("docs/final_fixed_goal.csv")))
        rows = [r for case in range(4101,4113) for r in trial(case,abc,"replay",scene)]
        for r, old in zip(rows,archive,strict=True):
            assert (r["case"],r["arm"]) == (int(old["case"]),old["arm"])
            assert r["final_error"] == float(old["final_error"])
            assert str(r["end_settled"]) == old["settled_at_end"]
        dev = []
        for i, dx in enumerate(OFFSETS):
            for case in range(5001,5007):
                dev += trial(case,{f"D{i}":dx},"dev",scene)
        best = min(OFFSETS,key=lambda dx: np.mean([r["final_error"] for r in dev if r["offset_m"]==dx]))
        freeze = dict(ordinary_calibrated_offset_m=best, goal_m=GOAL,
                      dev=list(range(5001,5007)),held=list(range(5101,5113)), options_m=OFFSETS)
        (ROOT/"calibration.json").write_text(json.dumps(freeze,indent=2))
        abc["D"] = best
        held = [r for case in freeze["held"] for r in trial(case,abc,"held",scene)]
        free = trial(4101,abc,"free",scene,unobstructed=True)
        assert all(r["contact_s"] == 0 for r in free)
        save(Path("docs/supplement_push.csv"), rows+dev+held+free)


def heavier():
    # Transfer the published 1 kg schedules unchanged to paired 1.5 kg placements.
    freeze = json.loads((ROOT/"calibration.json").read_text())
    offsets = {"A":0., "B":0., "C":.06, "D":freeze["ordinary_calibrated_offset_m"]}
    with TemporaryDirectory(dir=ROOT) as tmp:
        rows = []
        for split,cases in [("dev",freeze["dev"][:2]),("held",freeze["held"])]:
            for case in cases:
                rows.extend(trial(case,offsets,"heavy_"+split,Path(tmp)/"box.mjb",mass=1.5))
        save(Path("docs/supplement_push_heavy.csv"),rows)


if __name__ == "__main__":
    if sys.argv[1:] == ["heavy"]:
        heavier()
    else:
        main()
