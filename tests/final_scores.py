"""After all timed runs: paired scene-cluster summaries and executed challenge moments."""
import csv
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from cone_contact_wrench import analyze
from final_scenes import ROOT, MANIFEST
from final_trials import digest, FREEZE


def write(path, rows):
    with Path(path).open("w") as f:
        w = csv.DictWriter(f, list(dict.fromkeys(k for r in rows for k in r)), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def interval(ids, values):
    means = np.array([np.mean(np.array(values)[np.array(ids)==i]) for i in sorted(set(ids))])
    rng = np.random.default_rng(20261124)
    draws = rng.choice(means, (10000,len(means)),replace=True).mean(1)
    return np.percentile(draws,[2.5,97.5]).tolist()


def main():
    keys = ("block", "task", "physics", "arm")
    rows, latency = [], defaultdict(list)
    warm = {}
    for p in sorted(ROOT.glob("*/*/final.json")):
        r = json.loads(p.read_text())
        assert r.pop("lock")==digest(FREEZE)
        task = r["task"]
        folder = p.parent
        with np.load(folder/"episode.npz") as e:
            if r["block"]=="sumo":
                lat = e["planning_s"]
                assert int(e["calls"].sum()) == r["calls"]
                loads = e["substep_forces"]
                f = loads[:,:,3:6].mean(1)
                r["signed_hand_impulse_xyz_Ns"] = (.02 * f.sum(0)).tolist()
                r["end_object_speed_m_s"] = float(np.linalg.norm(e["qvel"][-1,-6:-3]))
                if task=="cone":
                    r["end_angular_rad_s"] = float(np.linalg.norm(e["qvel"][-1,-3:]))
                    r["end_hold_hand_ticks"] = int(loads[-25:,:,9].any(1).sum())
                    r["end_hold_other_Ns"] = float(loads[-25:,:,10].sum())
                    q = e["qpos"]
                    initial = e["initial"][-4:]
                    a0 = np.degrees(np.arccos(np.clip(1-2*np.sum(initial[1:3]**2),-1,1)))
                    ang = np.degrees(np.arccos(np.clip(1-2*(q[:,-3]**2+q[:,-2]**2),-1,1)))
                    i = np.flatnonzero((a0-ang >= 10.) & np.maximum.accumulate(loads[:,:,9].any(1)))
                    r["first_progress_s"] = (i[0]+1)*.02 if len(i) else None
            elif task=="box":
                with np.load(folder/"plans.npz") as plans:
                    lat = plans["planning_s"]
                    r["valid_fraction"] = plans["valid_candidates"].sum()/(24*r["updates"])
            elif task=="cone":
                lat = e["planning_s"]
                r["valid_fraction"] = e["valid_candidates"].sum()/(16*r["updates"])
                r["end_angular_rad_s"] = float(np.linalg.norm(e["qvel"][-1,-3:]))
                r["end_hold_hand_ticks"] = int(e["gripper_touch"][50:][-25:].sum())
                r["end_hold_other_Ns"] = float(e["other_robot_cone_impulse_Ns"][50:][-25:].sum())
                q = e["qpos"][49:]
                ang = np.degrees(np.arccos(np.clip(1-2*(q[:,-3]**2+q[:,-2]**2),-1,1)))
                i = np.flatnonzero((ang[0]-ang[1:] >= 10) & np.maximum.accumulate(e["gripper_touch"][50:]))
                r["first_progress_s"] = (i[0]+1)*.02 if len(i) else None
                if r["block"]=="challenge":
                    moment = analyze(folder,trace=e,start=50)
                    r.update(tipping_Nms=moment["supported_tipping_Nms"],
                             axial_m=moment["impulse_weighted_axial_m"],
                             supported_hand_Ns=moment["supported_hand_Ns"])
            else:
                lat = e["latency_s"]
                r["valid_fraction"] = e["valid_count"].sum()/(24*r["updates"])
                r["interference_at_stop"] = r.pop("rejection").get("cable_crossing",0)
                r["settled_s"] = r["sim_s"] if r["strict"] else None
            if r["block"]!="sumo" and task!="tether":
                f = e["gripper_on_"+task+"_N"][50:]
                r["end_object_speed_m_s"] = float(np.linalg.norm(e["qvel"][-1,-6:-3]))
                r["signed_hand_impulse_xyz_Ns"] = (.02*f.sum(0)).tolist()
                if task=="box":
                    r["cycle_p95_ms"] = np.percentile(e["cycle_s"]*1000,95)
            latency[tuple(r[k] for k in keys)].extend(1000*lat)
            if r["block"]!="sumo":
                if task=="tether":
                    w = e["initial_qpos"]
                else:
                    w = np.column_stack([e[k][:50] for k in ("qpos","qvel","action","policy_command")])
                np.testing.assert_array_equal(warm.setdefault(r["scene_seed"],w),w)
        rows.append(r)
    rows.sort(key=lambda r:tuple(r[k] for k in ("block","task","scene_seed","cem_seed","physics","arm")))
    write("docs/final_trials.csv",rows)
    grouped = defaultdict(list)
    for r in rows:
        grouped[tuple(r[k] for k in keys)].append(r)
    totals = []
    for key, group in grouped.items():
        lat = np.array(latency[key])
        ids = [r["scene_seed"] for r in group]
        success = [int(r["strict"]) for r in group]
        err = [r["final_error"] for r in group]
        lo, hi = interval(ids,success)
        p50,p95,peak = np.percentile(lat,[50,95,100])
        total = dict(zip(keys,key), scenes=len(set(ids)), episodes=len(group), success=sum(success),
                     success_ci_lo=lo,success_ci_hi=hi, mean_error=np.mean(err), median_error=np.median(err),
                     upright=sum(r.get("upright_s") is not None for r in group) if key[1]=="cone" else "",
                     safe_success=sum(r.get("safe_strict",r["strict"]) for r in group),
                     p50_ms=p50,p95_ms=p95,max_ms=peak,updates=len(lat),
                     misses_100ms=int(np.sum(lat>100)), period_misses=sum(r["period_misses"] for r in group),
                     completed_calls=sum(r["calls"] for r in group),scheduled_calls=sum(r["scheduled"] for r in group),
                     unsafe_episodes=sum((r.get("other_Ns") or 0)>0 for r in group),
                     max_tilt_deg=max(r["max_tilt_deg"] or 0 for r in group))
        total["joint_progress_count"] = (sum(r.get("first_progress_s") is not None for r in group)
                                         if key[1]=="cone" else "")
        total["no_valid_plan"] = sum(r["status"]=="no_valid_plan" for r in group)
        total["mean_observed_s"] = np.mean([r["sim_s"] for r in group])
        total["released_without_other_in_hold"] = (sum(r["strict"] and r["end_hold_other_Ns"]==0
                                                       for r in group) if key[1]=="cone" else "")
        for k in ["hand_impulse_Ns","hand_tick_peak_N","contact_s","virtual_mean_N","valid_fraction",
                  "tension_mean_N","tension_peak_N","taut_entries","tipping_Nms","axial_m","prefix_xy_rms_mm"]:
            data = [r[k] for r in group if r.get(k) is not None]
            if data:
                total["avg_"+k]=np.mean(data)
        totals.append(total)
    write("docs/final_groups.csv",totals)
    paired = []
    for a,b in combinations(grouped,2):
        same = a[:3]==b[:3]
        knowledge = a[0:2]==b[0:2] and a[3]==b[3] and a[2]!=b[2]
        direct = a[1]==b[1] and {a[0],b[0]}=={"general","sumo"}
        if not (same or knowledge or direct):
            continue
        by_pair = {(r["scene_seed"],r["cem_seed"]):r for r in grouped[b]}
        pairs = [(r,by_pair[(r["scene_seed"],r["cem_seed"])]) for r in grouped[a]
                 if (r["scene_seed"],r["cem_seed"]) in by_pair]
        if not pairs:
            continue
        ids = [x["scene_seed"] for x,y in pairs]
        wins = sum(x["strict"] and not y["strict"] for x,y in pairs)
        losses = sum(y["strict"] and not x["strict"] for x,y in pairs)
        jointly = [x["sim_s"]-y["sim_s"] for x,y in pairs if x["strict"] and y["strict"]]
        sd = [int(x["strict"])-int(y["strict"]) for x,y in pairs]
        ed = [x["final_error"]-y["final_error"] for x,y in pairs]
        native_times = [x["upright_s"]-y["upright_s"] for x,y in pairs
                        if x.get("upright_s") is not None and y.get("upright_s") is not None]
        paired.append(dict(first="/".join(a),second="/".join(b),pairs=len(pairs),scenes=len(set(ids)),
                           first_only=wins,second_only=losses,jointly_solved=len(jointly),
                           success_delta=np.mean(sd),success_delta_ci=interval(ids,sd),
                           error_delta=np.mean(ed),error_delta_ci=interval(ids,ed),
                           joint_time_delta_s=np.mean(jointly) if jointly else None,
                           jointly_upright=len(native_times) if a[1]=="cone" else None,
                           joint_upright_time_delta_s=np.mean(native_times) if native_times else None))
    write("docs/final_pairs.csv",paired)
    manifest = list(csv.DictReader(MANIFEST.open()))
    expected = sum((10 if r["task"]=="cone" else 6) if r["block"]=="general" else 12 for r in manifest)
    expected += 4*sum(r["task"]!="tether" and r["block"]=="general" for r in manifest)+4
    assert len(rows)==expected, (len(rows),expected)
    print(len(rows), "complete episodes;", len(grouped), "groups;", len(paired), "paired contrasts")


if __name__=="__main__":
    main()
