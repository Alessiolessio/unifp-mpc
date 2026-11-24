"""Score strict outcomes and physical wrench replay, including every unsuccessful episode."""
import csv
import json
from pathlib import Path

import numpy as np

from cone_contact_wrench import analyze
from kinematics import angle_deg, nominal_targets


def measure(folder):
    s = json.loads((folder/"summary.json").read_text())
    seed, treatment = folder.name.split("-",1)
    treatment, sampler = treatment.rsplit("-",1)
    with np.load(folder/"episode.npz") as e:
        q = e["qpos"][49:]
        tilt = angle_deg(q[:, -4:])
        # A 10 degree improvement after the first active gripper contact.
        touched = np.maximum.accumulate(e["gripper_touch"][50:])
        progress = np.flatnonzero(touched & (tilt[0]-tilt[1:] >= 10))
        u = e["requests"].reshape(-1,e["requests"].shape[-1])
        saturation = np.linalg.norm(nominal_targets(q[:-1,:7],u[:,:3])-u[:,:3],axis=1)
        lat = 1000*e["planning_s"]
        wrench = analyze(folder, trace=e, start=50)
        return dict(scene=s["scenario"]["scenario_seed"], cem=int(seed), treatment=treatment,
                    sampler=sampler,status=s["status"], final_deg=tilt[-1], min_deg=tilt.min(),
                    productive_s=(1+progress[0])*.02 if len(progress) else None,
                    first_upright_s=s["sumo_first_s"], control_s=s["control_s"],
                    valid=s["valid_candidates"]/(16*len(lat)), clipped=np.mean(saturation>1e-5),
                    contact_s=.02*e["contact_duty"][50:].sum(),
                    axial_cm=(100*wrench["impulse_weighted_axial_m"]
                              if wrench["impulse_weighted_axial_m"] is not None else None),
                    support_moment_Nms=wrench["supported_tipping_Nms"],
                    **s["planning"],updates=len(lat))


if __name__ == "__main__":
    paths=sorted(Path("artifacts/exploration").glob("*/*/summary.json"))
    with Path("docs/exploration_trials.csv").open("w") as f:
        for i,path in enumerate(paths):
            row=measure(path.parent)
            if i==0:
                writer=csv.DictWriter(f,fieldnames=row,lineterminator="\n")
                writer.writeheader()
            writer.writerow(row)
