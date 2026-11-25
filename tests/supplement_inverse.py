"""Feasibility screen for treating archived raw requests as effective-target proposals."""
from pathlib import Path

import numpy as np

from final_policy import save
from kinematics import nominal_targets, world_force


def main():
    rows = []
    for task,ids in [("box",["00","06"]),("cone",["00","02"])]:
        for case in ids:
            path = Path("artifacts/uncertainty")/task/case/"conditioned/episode.npz"
            with np.load(path) as e:
                u, q = e["forecast_request"], e["forecast_qpos"]
                pre = np.concatenate((e["forecast_pre_root"][:,None,:7],q[:,:-1,:7]),axis=1)
                wished = u[:,:,:3]
                if task=="box":
                    f = np.concatenate((u[:,:,6:8],np.zeros_like(u[:,:,0:1])),axis=-1)
                else:
                    f = world_force(pre,u)
                # Putative inverse from projected effective point: an infeasible endpoint
                # proves shortening is needed. A feasible endpoint may still cross a boundary.
                inv = nominal_targets(pre,wished)-f/200
                bad = np.linalg.norm(inv-nominal_targets(pre,inv),axis=-1) > 1e-5
                # Independent knot-wise E and F ramps need to respect the existing world
                # nominal-request speed bound even before state-specific spherical projection.
                raw_inv = wished-f/200
                speed = np.linalg.norm(np.diff(raw_inv,axis=1),axis=-1)/.02
                raw_speed = np.linalg.norm(np.diff(wished,axis=1),axis=-1)/.02
                assert raw_speed.max() < .501
                rows.append(dict(task=task,path=str(path),forecast_ticks=bad.size,
                                 endpoint_outside=bad.sum(),intervals=speed.size,
                                 inverse_over_slew=(speed>.5001).sum(),raw_max_m_s=raw_speed.max(),
                                 inverse_max_m_s=speed.max()))
    save(Path("docs/supplement_inverse.csv"),rows)


if __name__=="__main__":
    main()
