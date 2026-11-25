"""Trace-level coordination and paired differences on original and fresh fixed pushes."""
import csv
from pathlib import Path

import mujoco
import numpy as np

from final_scores import write
from fixed_policy_loads import rms, length
from prepare_scene import JOINTS
from supplement_push import ROOT


def main():
    m = mujoco.MjModel.from_binary_path("artifacts/spot_box.mjb")
    joints = [s for s in JOINTS if s.startswith("arm_")]
    actor = [JOINTS.index(s) for s in joints]
    q_indices = [m.joint(s).qposadr.item() for s in joints]
    rows = []
    measurements = list(csv.DictReader(open("docs/supplement_push.csv")))
    heavy = Path("docs/supplement_push_heavy.csv")
    if heavy.exists():
        measurements += list(csv.DictReader(heavy.open()))
    for cohort in sorted({r["cohort"] for r in measurements} & {"replay","held","heavy_held"}):
        controls = "AC" if cohort=="replay" else "ACD"
        for case in sorted({r["case"] for r in measurements if r["cohort"]==cohort}):
            by_arm = {r["arm"]:r for r in measurements if r["cohort"]==cohort and r["case"]==case}
            scores = by_arm["B"]
            folder = ROOT/cohort/case
            for arm in controls:
                ref = by_arm[arm]
                with np.load(folder/"B.npz") as b,np.load(folder/f"{arm}.npz") as c:
                    loaded = np.flatnonzero((b["contact_duty"]+c["contact_duty"])>0)
                    t = int(loaded[0]) if loaded.size else len(b["action"])
                    act = b["action"]-c["action"]
                    angles = b["qpos"][:,q_indices]-c["qpos"][:,q_indices]
                    offset = b["ee_world"]-c["ee_world"]
                    r = dict(cohort=cohort,case=case,ref=arm,
                             force_only=scores["end_settled"]=="True" and ref["end_settled"]!="True",
                             ref_only=scores["end_settled"]!="True" and ref["end_settled"]=="True",
                             first_loaded_tick=t if loaded.size else None,precontact_finger_rms_m=rms(length(offset[:t])),
                             hand_rms_m=rms(length(offset)),precontact_action_arm_rms=rms(act[:t,actor]),
                             base_xy_rms_m=rms(length(b["qpos"][:,:2]-c["qpos"][:,:2])))
                    for j,n in enumerate(joints):
                        r["first_action_delta_"+n] = act[0,actor[j]]
                        r["angle_rms_rad_"+n] = rms(angles[:,j])
                    for k in ["final_error","effective_rms_mm","progress_m","impulse_x_Ns",
                              "abs_lateral_impulse_Ns","contact_s","cross_track_m","peak_forward_N"]:
                        r["delta_"+k] = float(scores[k])-float(ref[k])
                    rows.append(r)
    write("docs/supplement_push_pairs.csv",rows)


if __name__ == "__main__":
    main()
