"""Verify recorded native pilot forces and reproducible optimizer decisions."""
import csv
import json
from pathlib import Path

import numpy as np

root = Path("artifacts/final_dev_confirm")
rows = []
for p in sorted(root.glob("*/*/*/result.json")):
    r = json.loads(p.read_text())
    task, case, count = p.parent.relative_to(root).parts
    e = np.load(p.parent/"episode.npz")
    earlier = np.load(Path("artifacts/final_dev")/task/case/count/"episode.npz")
    n = min(len(e["qpos"]),len(earlier["qpos"]))
    for key in ("action","qpos","qvel"):
        np.testing.assert_array_equal(e[key][:n],earlier[key][:n])
    row = dict(task=task, pilot=case, count=r["candidates"], period_ms=r["period_ms"],
               compared_ticks=n, **{k:r.get(k) for k in ("status","strict","safe_strict","upright_s",
                        "final_error","contact_s","hand_impulse_Ns","other_Ns","p50_ms","p95_ms",
                        "max_ms","period_misses","misses_100ms","calls","scheduled")})
    if task=="box":
        f = e["substep_forces"][:,:,:9].reshape(-1,2,3,3).sum(2).mean(1)
        f[:,2] -= 9.81
        dv = np.diff(e["qvel"][:,-6:-3],axis=0)
        residual = np.linalg.norm(.02*f[1:]-dv,axis=1)
        row["max_momentum_residual_Ns"] = residual.max()
        assert residual.max() < .0012
    rows.append(row)
with Path("docs/final_pilots.csv").open("w") as f:
    w = csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)),lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
print(len(rows), "pilot assignments validated")
