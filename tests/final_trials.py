"""Serial, resumable execution of the frozen final manifest. Run from repository root."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import platform
import importlib.metadata
from types import SimpleNamespace

import numpy as np
from box_mpc import run
from check_common_score import evaluate
from cone_mpc import run_cone
from final_pilots import relic
from final_scenes import MANIFEST, POLICY, ROOT
from tether_mpc import episode
from unifp_native import Rollouts, World

FREEZE = Path("docs/final_lock.json")
SEEDS = (311,337)


def digest(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def fingerprint():
    files = [MANIFEST, Path(POLICY), Path("pixi.lock"), Path("artifacts/spot_contact.npz"),
             *Path("build").glob("unifp_native*.so"), Path("build/sumo_observer.so"),
             Path("../sumo/.judo-src/judo/models/policies/spot_locomotion.onnx"),
             *map(Path, ["tests/sumo_common.py", "tests/sumo_observer.c", "tests/final_trials.py",
                         "tests/final_scenes.py", "tests/final_pilots.py", "tests/check_common_score.py",
                         "docs/frozen_protocol.md"]),
             *Path("tools").glob("*.py"), *Path("native").glob("*.h"), Path("native/world.cpp")]
    installed = Path("../sumo/.pixi/envs/default/lib/python3.13/site-packages")
    files += [installed/"judo/models/policies/spot_locomotion.onnx", *installed.glob("mujoco_extensions/policy_rollout/*.so"),
              installed/"mujoco/libmujoco.so.3.6.0", Path("../sumo/pixi.lock"), Path(".pixi/envs/default/lib/libonnxruntime.so.1.24.2"),
              Path("../sumo/.pixi/envs/default/lib/libonnxruntime.so.1.24.2")]
    return dict(cpu=subprocess.check_output(["lscpu"],text=True).split("Model name:")[1].splitlines()[0].strip(),
                python=platform.python_version(), packages={k:importlib.metadata.version(k) for k in
                                                            ("numpy","mujoco")},
                sha256={str(p):digest(p) for p in files}, affinity=sorted(os.sched_getaffinity(0)),
                revisions={repo:subprocess.check_output(["git","-C",repo,"rev-parse","HEAD"],text=True).strip()
                           for repo in [".","../sumo","../sumo/.judo-src","../judo","../unifp-spot"]})


class CountPool:
    def __init__(self, pool):
        self.pool = pool
        self.mismatched = pool.mismatched
        self.completed = []
        self.scheduled = []

    def evaluate(self, snapshot, commands, **kwargs):
        batch = self.pool.evaluate(snapshot, commands, **kwargs)
        self.completed.append(int(batch["valid_steps"].sum()))
        self.scheduled.append(int(np.prod(commands.shape[:2])))
        return batch


def unifp(row, seed, arm, informed, scene, plant, out):
    task = row["task"]
    pool = Rollouts(World(str(plant if informed else scene), POLICY), 8)
    live = pool.execution_world(1. if informed else float(row["mass_scale"]))
    pool = CountPool(pool)
    inputs = json.loads(row["scenario"])
    force = "off" if arm=="A" else ("conditioned" if arm.startswith("B") else "matched")
    seconds = float(row["seconds"])
    out.mkdir(exist_ok=True, parents=True)
    if not (out/"scene.mjb").exists():
        (out/"scene.mjb").symlink_to(os.path.relpath(plant, out))
    if task=="tether":
        summary = episode(live,pool,force,np.array(inputs["goal_offset_m"]),inputs["limit_m"],
                          seed,seconds,out,plant,row["scene_seed"])
        score = dict(strict=summary["status"]=="settled", final_error=summary["final_error_m"],
                     sim_s=summary["simulated_s"], max_tilt_deg=summary.get("tilt_max_deg"),
                     **{k:summary.get(k) for k in ("tension_mean_N","tension_peak_N","taut_entries",
                         "cross_track_max_m","body_box_impulse_Ns","virtual_mean_N","loaded_swing_deg")})
        timing = {k:summary[k] for k in ("p50_ms","p95_ms","max_ms","misses_100ms")}
    else:
        args = SimpleNamespace(controller="mobile", workers=8, iterations=1, warmup_ee=None, out=out,
                               ee_force=force, force_axes="xyz" if arm.endswith("xyz") else "z",
                               objective="upright_buffer", sumo_sampling=True, horizon=1., seed=seed,
                               candidates=16 if task=="cone" else 24, seconds=seconds, policy=POLICY)
        if task=="cone":
            summary = run_cone(args,out,inputs,live,pool)
        else:
            summary = run(args,scenario=scene.parent,live=live,pool=pool,quiet=True)
        summary["scene"] = str(plant)
        (out/"summary.json").write_text(json.dumps(summary,indent=2))
        with np.load(out/"episode.npz") as e:
            score = evaluate(task,e,summary,live.object_mass)
            assert score["strict"] == (summary["status"]=="success")
            ticks = len(e["qpos"])-50
            hand = np.linalg.norm(e[f"gripper_on_{task}_N"][50:],axis=1)
            score.update(sim_s=ticks*.02, hand_tick_peak_N=float(hand.max(initial=0)),
                         hand_impulse_Ns=float(.02*hand.sum()), contact_s=float(.02*e["contact_duty"][50:].sum()),
                         virtual_mean_N=float(np.linalg.norm(e["policy_command"][50:,6:9],axis=1).sum()/max(ticks,1)))
            if task=="cone":
                score.update(min_gap_m=summary["min_chassis_gap_m"],
                             translation_m=float(np.linalg.norm(summary["displacement_com_m"][:2])))
            if "forecast_qpos" in e and row["block"]=="challenge":
                p = e["forecast_qpos"][:,:5].reshape(-1,e["qpos"].shape[1])
                diff = p[4::5,-7:-5]-e["qpos"][54::5,-7:-5]
                score["prefix_xy_rms_mm"] = float(1000*np.sqrt(np.mean((diff**2).sum(1))))
                f = e["forecast_gripper_on_"+task+"_N"][:,4]
                score["prefix_hand_rms_N"] = float(np.sqrt(np.mean(((f-e["gripper_on_"+task+"_N"][54::5])**2).sum(1))))
        timing = summary["planning"]
    result = dict(**score, **timing, status=summary["status"], updates=len(pool.completed),
                  calls=sum(pool.completed), scheduled=sum(pool.scheduled), period_ms=100,
                  period_misses=timing["misses_100ms"])
    if task=="tether":
        result["rejection"] = summary["rejected_last_batch"]
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("block",choices=["lock","general","challenge","sumo"])
    args = p.parse_args()
    state = fingerprint()
    if args.block=="lock":
        with FREEZE.open("x") as f:
            json.dump(state,f,indent=2)
        return
    if json.loads(FREEZE.read_text()) != state:
        raise RuntimeError("Source, binary, manifest, or CPU affinity differs from frozen run")
    identity = digest(FREEZE)
    for row in csv.DictReader(MANIFEST.open()):
        task, scene_seed = row["task"], int(row["scene_seed"])
        cross = args.block=="sumo"
        if row["block"] != ("general" if cross else args.block) or (cross and task=="tether"):
            continue
        folder = ROOT/str(scene_seed)
        scene = folder/"scene.mjb"
        plant = scene if row["block"]=="general" else folder/"execution.mjb"
        assert digest(scene)==row["model_sha256"] and digest(plant)==row["plant_sha256"]
        for seed in SEEDS:
            arms = ["A","B","C"]
            if task=="cone":
                arms = ["A","Bz","Cz"] + (["Bxyz","Cxyz"] if args.block=="general" else [])
            if cross:
                arms = ["native","work"]
                if seed==311 and scene_seed in [3001,3013,3101,3109]:
                    arms += ["runtime"]
            for informed in ([False,True] if args.block=="challenge" else [False]):
                for arm in arms:
                    setting = ("informed" if informed else "blind") if args.block=="challenge" else "known"
                    out = folder/f"{seed}-{setting}-{arm}"
                    done = out/"final.json"
                    if done.exists():
                        assert json.loads(done.read_text())["lock"] == identity
                        continue
                    if cross:
                        count = {"native":0,"work":6 if task=="box" else 8,"runtime":16}[arm]
                        relic(task,scene,out,seed,float(row["seconds"]),count,
                              json.loads(row["goal_xy"]) if task=="box" else None)
                        result = json.loads((out/"result.json").read_text())
                        result.pop("initial")
                    else:
                        result = unifp(row,seed,arm,informed,scene,plant,out)
                    result.update(block=args.block, task=task, scene_seed=scene_seed, tier=row["tier"],
                                  cem_seed=seed, arm=arm, physics=setting, lock=identity)
                    tmp = out/"final.tmp"
                    tmp.write_text(json.dumps(result,indent=2))
                    tmp.replace(done)
                    print(args.block, task, scene_seed, seed, arm, setting, result["status"],
                          result["final_error"],flush=True)


if __name__ == "__main__":
    main()
