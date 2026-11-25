"""Audit the observed ReLIC contact histories and pair the supplemental runtime cohort."""
import csv
import json
from pathlib import Path

import numpy as np

from final_scores import write
from physical_score import tilt, score
from supplement_sumo import ROOT, LOCK


def release(q,v,loads,mass):
    hand = loads[:,:,9].any(1)
    other = loads[:,:,10].sum(1)
    a = 1-2*(q[:,-3]**2+q[:,-2]**2)
    hits = np.flatnonzero(a>=.9)
    start = hits[0] if len(hits) else len(q)
    stable = (tilt(q)<30) & (q[:,2]>=.35)
    return dict(upright_ticks=len(hits), hand_s_after_first_upright=.02*hand[start:].sum(),
                early_other_Ns=other[:-25].sum(), final_other_Ns=other[-25:].sum(),
                post_upright_other_Ns=other[start:].sum(),hand_ticks_final=hand[-25:].sum(),
                poor_tilt_ticks_final=(a[-25:] < np.cos(np.deg2rad(10))).sum(),
                fast_ticks_final=(np.linalg.norm(v[-25:,-3:],axis=1)>.25).sum(),
                unsupported_ticks_final=(loads[-25:,:,2].mean(1)<.5*9.81*mass).sum(),
                unstable_ticks_final=(~stable[-25:]).sum(),unstable_ticks_total=(~stable).sum(),
                upright_s=.02*(start+1) if len(hits) else None)


def main():
    manifest = {r["scene_seed"]:r for r in csv.DictReader(open("docs/final_scenes.csv"))}
    old = list(csv.DictReader(open("docs/final_trials.csv")))
    trials, cone, groups, paired = [], [], [], []
    latency = {"box":[],"cone":[]}
    cycles = {"box":[],"cone":[]}
    selected = json.loads(LOCK.read_text())["selected"]
    for run in sorted((ROOT/"paired").glob("*/*/assignment.json")):
        folder = run.parent
        case,seed = folder.parent.name, int(folder.name)
        spec = json.loads(run.read_text())
        r = json.loads((folder/"result.json").read_text())
        task = spec["task"]
        row = manifest[case]
        assert [spec["count"],spec["horizon"]] == selected[task]
        assert np.array_equal(r.pop("initial"), json.loads((Path("artifacts/final")/case/"warm.json").read_text()))
        with np.load(folder/"episode.npz") as e:
            loads = e["substep_forces"]
            q,v = e["qpos"],e["qvel"]
            rescored = score(task,q,v,loads[:,:,9].any(1),loads[:,:,2].mean(1),
                             loads[:,:,10].sum(1),float(row["nominal_kg"]),e["initial"],np.array(spec["goal"]))
            for k in rescored:
                assert r[k] == rescored[k]
            if task=="cone":
                cone.append(dict(scene=case,seed=seed,arm="supplement",strict=r["strict"],
                                 status=r["status"],**release(q,v,loads,3.36)))
            cycle = np.load(folder/"cycle_s.npy")*1000
            plan = e["planning_s"]*1000
            assert (cycle>=plan).all() and len(cycle)==len(plan)==r["updates"]
            latency[task].extend(plan)
            cycles[task].extend(cycle)
            assert r["calls"] == e["calls"].sum() and r["period_ms"]==100
            r.update(task=task,scene=case,tier=row["tier"],seed=seed,**{"horizon":spec["horizon"]},
                     rollout_physics_steps=2*r["calls"],execution_physics_steps=2*len(q),
                     cycle_p95_ms=np.percentile(cycle,95),cycle_max_ms=cycle.max(),
                     cycle_misses=int((cycle>100).sum()),
                     hand_impulse_world_xyz_Ns=(.01*loads[:,:,3:6].sum((0,1))).tolist())
        trials.append(r)
    assert len(trials)==54 and len({(r["scene"],r["seed"]) for r in trials})==54
    for r in old:
        if r["block"] != "sumo" or r["task"] != "cone" or r["arm"] not in ("native","work"):
            continue
        folder = Path("artifacts/final")/r["scene_seed"]/(r["cem_seed"]+"-known-"+r["arm"])
        with np.load(folder/"episode.npz") as e:
            cone.append(dict(scene=r["scene_seed"],seed=r["cem_seed"],arm=r["arm"],
                             strict=r["strict"]=="True",status=r["status"],
                             **release(e["qpos"],e["qvel"],e["substep_forces"],3.36)))
    for task in latency:
        t = [r for r in trials if r["task"]==task]
        ms = np.array(latency[task])
        cycle = np.array(cycles[task])
        groups.append(dict(task=task,episodes=len(t),strict=sum(r["strict"] for r in t),
                           clean=sum(r["safe_strict"] for r in t),
                           upright=sum(r.get("upright_s") is not None for r in t) if task=="cone" else None,
                           error=np.mean([r["final_error"] for r in t]),p50_ms=np.median(ms),
                           p95_ms=np.percentile(ms,95),max_ms=ms.max(),misses=(ms>100).sum(),
                           updates=len(ms),cycle_p95_ms=np.percentile(cycle,95),cycle_max_ms=cycle.max(),
                           cycle_misses=(cycle>100).sum(), calls=sum(r["calls"] for r in t),
                           scheduled=sum(r["scheduled"] for r in t), max_tilt=max(r["max_tilt_deg"] for r in t),
                           unwanted=sum(r["other_Ns"]>0 for r in t),
                           mean_impulse_Ns=np.mean([r["hand_impulse_Ns"] for r in t])))
        for arm in (["A","B","C","native","work"] if task=="box" else
                    ["A","Bz","Cz","Bxyz","Cxyz","native","work"]):
            ref = {(r["scene_seed"],int(r["cem_seed"])):r for r in old if r["task"]==task
                   and r["arm"]==arm and r["block"] in ("general","sumo")}
            twins = [ref[(r["scene"],r["seed"])] for r in t]
            a,b = np.array([r["strict"] for r in t]),np.array([r["strict"]=="True" for r in twins])
            joint = [x["settled_s"]-float(y["settled_s"]) for x,y in zip(t,twins)
                     if x["strict"] and y["strict"]=="True"]
            paired.append(dict(task=task,reference=arm,runtime_only=(a&~b).sum(),reference_only=(b&~a).sum(),
                               jointly=len(joint),mean_joint_delta_s=np.mean(joint) if joint else None,
                               mean_error_delta=np.mean([x["final_error"]-float(y["final_error"])
                                                         for x,y in zip(t,twins)])))
    pilots = []
    for case in ("near-0","general-82"):
        for reward,folder in [("registered", ROOT/"dev/cone"/case/"24x2"),
                              ("release", ROOT/"release_dev"/case)]:
            result = json.loads((folder/"result.json").read_text())
            spec = json.loads((folder/"assignment.json").read_text())
            with np.load(folder/"episode.npz") as e:
                hold = release(e["qpos"],e["qvel"],e["substep_forces"],3.36)
            pilots.append(dict(case=case,objective=reward,source=str(folder),seed=spec["seed"],
                               **{k:v for k,v in result.items() if k not in ("initial","upright_s")},
                               **hold))
    write("docs/supplement_release_dev.csv",pilots)
    for name,data in [("runtime",trials),("release",cone),("runtime_groups",groups),("runtime_pairs",paired)]:
        write(f"docs/supplement_{name}.csv",data)


if __name__=="__main__":
    main()
