"""Calibrate on archived development scenes, freeze, then run the 54 paired assignments."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter

import numpy as np

ROOT = Path("artifacts/supplemental/sumo")
LOCK = ROOT / "runtime_lock.json"
OPTIONS = {"box": [(20, 2), (40, 1)], "cone": [(24, 2), (48, 1)]}
TARGET = {"box": 43., "cone": 61.}
DEV = {"box": ["00", "08"], "cone": ["near-0", "general-82"]}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker(spec, out):
    import sumo_common as s
    task = spec["task"]
    name = "spot_box_push" if task == "box" else "spot_cone_upright"
    s.set_config_overrides(name, s.ControllerConfig, {"horizon": float(spec["horizon"])})
    # The common runner takes exactly two perf_counter readings around each plan.
    # Its external score is called once after that plan's five executed ticks.
    clock = []
    cycles = []
    original_score = s.score

    def timestamp():
        now = perf_counter()
        clock.append(now)
        return now

    def scored(*args):
        result = original_score(*args)
        cycles.append(perf_counter() - clock[-2])
        return result

    s.perf_counter, s.score = timestamp, scored
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "meshes").symlink_to(s.base.MODEL_PATH / "meshes", target_is_directory=True)
        s.base.MODEL_PATH = Path(tmp)
        s.run(spec["scene"], task, spec["seed"], spec["count"], spec["seconds"],
              np.asarray(spec["goal"]), out)
    with np.load(out/"episode.npz") as e:
        n = len(e["planning_s"])
        assert len(clock) == 2*n and len(cycles) == n
        assert np.all(e["calls"] <= spec["count"]*spec["horizon"]*50)
        np.testing.assert_array_equal(np.diff(np.asarray(clock).reshape(-1,2), axis=1)[:,0], e["planning_s"])
    np.save(out/"cycle_s.npy", cycles)
    (out/"assignment.json").write_text(json.dumps(spec, indent=2))


def launch(task, scene, out, seed, count, horizon, seconds, goal):
    spec = dict(task=task, scene=str(scene), seed=seed, count=count, horizon=horizon,
                seconds=seconds, goal=goal)
    if (out/"assignment.json").exists():
        assert spec == json.loads((out/"assignment.json").read_text())
        return
    subprocess.run(["../sumo/.pixi/envs/default/bin/python", __file__, "worker",
                    json.dumps(spec), str(out)], check=True, env=dict(os.environ,
                    OPENBLAS_NUM_THREADS="1", PYTHONPATH="tools", PYTHONDONTWRITEBYTECODE="1",
                    LD_PRELOAD=str(Path("build/sumo_observer.so").resolve())))


def development():
    rows = []
    selected = {}
    for task, cases in DEV.items():
        options = []
        for count, horizon in OPTIONS[task]:
            results, lat = [], []
            for case in cases:
                if task == "box":
                    path = Path("artifacts/uncertainty/box")/case
                    scene, summary = path/"nominal/scene.mjb", path/"off/summary.json"
                else:
                    path = Path("artifacts/cone_force")/case
                    scene, summary = path/"scene.mjb", sorted(path.glob("*-16-off/summary.json"))[0]
                s = json.loads(summary.read_text())
                out = ROOT/"dev"/task/case/f"{count}x{horizon}"
                launch(task,scene,out,s.get("seed", s.get("cem_seed")),count,horizon,
                       s["control_limit_s"],s.get("goal_xy_m"))
                results.append(json.loads((out/"result.json").read_text()))
                with np.load(out/"episode.npz") as e:
                    lat.extend(1000*e["planning_s"])
            ratio = abs(np.percentile(lat,95)/TARGET[task]-1)
            wins = sum(s["strict"] for s in results)
            clean = sum(s["safe_strict"] for s in results)
            error = np.mean([s["final_error"] for s in results])
            rows.append(dict(task=task,count=count,horizon=horizon,p95_ms=np.percentile(lat,95),
                             strict=wins, clean=clean, error=error))
            options.append(((max(0.,ratio-.20),-wins,-clean,error,ratio), [count,horizon]))
        selected[task] = min(options)[1]
    frozen = dict(selected=selected, development=rows, affinity=sorted(os.sched_getaffinity(0)),
                  frozen_lock=sha("docs/final_lock.json"), script=sha(__file__),
                  observer=sha("build/sumo_observer.so"))
    if LOCK.exists():
        assert json.loads(LOCK.read_text()) == frozen
    else:
        with LOCK.open("x") as f:
            json.dump(frozen,f,indent=2)


def benchmark():
    lock = json.loads(LOCK.read_text())
    assert lock["script"] == sha(__file__) and lock["observer"] == sha("build/sumo_observer.so")
    assert sorted(os.sched_getaffinity(0)) == lock["affinity"] == list(range(8))
    assert lock["frozen_lock"] == sha("docs/final_lock.json")
    for r in csv.DictReader(open("docs/final_scenes.csv")):
        task, case = r["task"], r["scene_seed"]
        if r["block"] != "general" or task not in OPTIONS:
            continue
        scene = Path("artifacts/final")/case/"scene.mjb"
        assert sha(scene) == r["model_sha256"]
        for seed in (311,337):
            launch(task,scene,ROOT/"paired"/case/str(seed),seed,*lock["selected"][task],
                   float(r["seconds"]),json.loads(r["goal_xy"]) if task == "box" else None)


if __name__ == "__main__":
    if sys.argv[1] == "worker":
        worker(json.loads(sys.argv[2]),Path(sys.argv[3]))
    elif sys.argv[1] == "dev":
        development()
    elif sys.argv[1] == "paired":
        benchmark()
