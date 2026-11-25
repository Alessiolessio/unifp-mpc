"""Check the completed supplemental evidence; write a portable SHA-256 inventory."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

import numpy as np

ROOT = Path("artifacts/supplemental")


def rows(name):
    with Path(f"docs/{name}.csv").open() as f:
        return list(csv.DictReader(f))


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def main():
    lock = Path("docs/final_lock.json")
    original = json.loads(lock.read_text())
    frozen = rows("final_trials")
    assert len(frozen) == len(list(Path("artifacts/final").glob("*/*/final.json"))) == 538
    verified_lock = digest(lock)
    for r in frozen:
        folder = Path("artifacts/final")/r["scene_seed"]/f"{r['cem_seed']}-{r['physics']}-{r['arm']}"
        assert json.loads((folder/"final.json").read_text())["lock"] == verified_lock
        assert (folder/"episode.npz").is_file()
    for p in ("artifacts/encoder_actor.onnx","build/sumo_observer.so","docs/final_scenes.csv",
              "tests/sumo_common.py","build/unifp_native.cpython-313-x86_64-linux-gnu.so"):
        assert digest(Path(p)) == original["sha256"][p]
    for protocol in [ROOT/"sumo/runtime_lock.json",ROOT/"uncertainty/protocol.json"]:
        data = json.loads(protocol.read_text())
        for p,expected in data.get("hashes",{}).items():
            assert digest(Path(p)) == expected
        if "script" in data:
            assert digest(Path("tests/supplement_sumo.py")) == data["script"]
    for r in rows("supplement_runtime"):
        d = ROOT/"sumo/paired"/r["scene"]/r["seed"]
        with np.load(d/"episode.npz") as tr, np.load(d/"kinematics.npz") as k:
            assert k["finger_world_m"].shape == (len(tr["qpos"]),3)
            assert tr["substep_forces"].shape == (len(tr["qpos"]),2,11)
            assert tr["calls"].sum() == int(r["calls"]) == int(r["scheduled"])
    mismatch = rows("supplement_mismatch")
    assert len(mismatch) == 72
    warm = {}
    for r in mismatch:
        d = ROOT/"uncertainty"/r["task"]/r["case"]/r["plant"]
        d /= ("informed" if r["informed"]=="True" else "nominal_prediction")
        d /= r["arm"]
        with np.load(d/"episode.npz") as e:
            pre = np.column_stack([e[k][:50] for k in ("qpos","qvel","action","policy_command")])
            np.testing.assert_array_equal(warm.setdefault((r["task"],r["case"],r["plant"]),pre),pre)
            assert len(e["forecast_start_tick"])*5 == int(r["ticks"])
            assert np.all(e["forecast_start_tick"] == 50+5*np.arange(len(e["forecast_start_tick"])))
    # The original calibrated binary and the separate supplemental build should
    # produce identical ordinary box commands and states at nominal parameters.
    paths = [Path("artifacts/uncertainty/box/00/off"),ROOT/"uncertainty/box/4/nominal/nominal_prediction/A"]
    with np.load(paths[0]/"episode.npz") as a,np.load(paths[1]/"episode.npz") as b:
        for k in ("qpos","qvel","action","policy_command","ee_world","gripper_on_box_N","forecast_request"):
            np.testing.assert_array_equal(a[k],b[k])
    counts = {"supplement_runtime":54,"supplement_release":72,"supplement_release_dev":4,
              "supplement_push":106,"supplement_push_heavy":56,"supplement_wall":42}
    for name,n in counts.items():
        assert len(rows(name)) == n, name
    for r in rows("supplement_push")+rows("supplement_push_heavy"):
        with np.load(ROOT/"push"/r["cohort"]/r["case"]/(r["arm"]+".npz")) as e:
            err = np.linalg.norm(e["qpos"][-25:,-7:-5]-e["goal_xy"],axis=1)
            speed = np.linalg.norm(e["qvel"][-5:,-6:-4],axis=1)
            assert np.isclose(err[-1],float(r["final_error"]),rtol=0,atol=1e-14)
            assert (err.max() <= .015 and speed.max() <= .02) == (r["end_settled"]=="True")
            np.testing.assert_array_equal(e["requested"],e["policy_command"])
    for p in ROOT.rglob("*.npz"):
        with zipfile.ZipFile(p) as z:
            assert z.testzip() is None, p
    ref_revs = {k:subprocess.check_output(["git","-C",k,"rev-parse","HEAD"],text=True).strip()
                for k in original["revisions"] if k!="."}
    assert all(original["revisions"][k]==v for k,v in ref_revs.items())
    version = dict(working_base=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                   frozen_lock_sha256=verified_lock, cpu=original["cpu"],affinity=list(range(8)),
                   versions=original["packages"] | {"onnxruntime":"1.24.2"}, python=original["python"],
                   checkpoint_sha256=original["sha256"]["artifacts/encoder_actor.onnx"],
                   reference_revisions=ref_revs)
    (ROOT/"provenance.json").write_text(json.dumps(version,indent=2))
    files = [p for p in ROOT.rglob("*") if p.is_file() and not p.is_symlink()
             and "build" not in p.parts and p.name!="evidence.sha256"]
    files += list((ROOT/"build").glob("unifp_native*.so"))
    files += list(Path("tests").glob("supplement*.py")) + [Path("tests/check_supplement.py")]
    files += list(Path("docs").glob("supplement*.csv")) + [lock,Path("native/world.cpp"),Path("native/rollouts.h")]
    (ROOT/"evidence.sha256").write_text("".join(f"{digest(p)}  {p}\n" for p in sorted(files)))
    print("538 frozen episode records, 54 paired ReLIC episodes, 72 mismatch runs, fixed-command outcomes,",
          "nominal parity, archive integrity and",len(files),"supplemental checksums verified.")


if __name__ == "__main__":
    main()
