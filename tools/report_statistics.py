"""Offline report tables and archive checks. Run from the repository root."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from physical_score import score

DOC = Path("docs")
OUT = DOC / "report_data"


def read(name):
    with (DOC / (name + ".csv")).open() as f:
        return list(csv.DictReader(f))


def write(name, rows):
    with (OUT / (name + ".csv")).open("w") as f:
        w = csv.DictWriter(f, rows[0], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def grouped(rows, keys):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)
    return groups


def stats(values):
    x = np.array(values, float)
    return dict(n=len(x), mean=x.mean(), sd=x.std(ddof=1) if len(x)>1 else "",
                min=x.min(), p05=np.percentile(x,5), median=np.median(x),
                p95=np.percentile(x,95), max=x.max(), rms=np.sqrt(np.mean(x*x)),
                mean_absolute=np.abs(x).mean())


def interval(ids, x):
    ids = np.array(ids)
    x = np.asarray(x, float)
    means = [x[ids==i].mean() for i in sorted(set(ids),key=int)]
    rng = np.random.default_rng(20261124)
    return np.percentile(rng.choice(means,(10000,len(means))).mean(1),[2.5,97.5])


def verify(row, source, scene, sub):
    task = row["task"]
    relic = sub or row["block"] == "sumo"
    goal = np.array(json.loads(scene["goal_xy"])) if scene["goal_xy"] else None
    with np.load(source / "episode.npz") as e:
        q, v = e["qpos"], e["qvel"]
        if task == "tether":
            error = np.linalg.norm(q[:, -7:-5] - goal, axis=1)
            slow = np.linalg.norm(v[:, -6:-4], axis=1) <= .08
            strict = any(((error <= .05) & slow)[i-25:i].all() for i in range(25,len(q)+1,5))
            np.testing.assert_allclose(error[-1],float(row["final_error"]),atol=1e-7)
            assert not e["cable_blocked"].any()
            assert not e["other_robot_box_impulse_Ns"].any()
            assert not e["gripper_on_box_N"].any()
            assert strict == (row["strict"]=="True")
            return
        if relic:
            initial = e["initial"]
            forces = e["substep_forces"]
            touch = forces[:,:,9].any(1)
            support = forces[:,:,2].mean(1)
            other = forces[:,:,10].sum(1)
        else:
            initial = q[49]
            q, v = q[50:], v[50:]
            touch = e["gripper_touch"][50:] if task=="cone" else e["contact_duty"][50:]>0
            support = e["ground_on_"+task+"_N"][50:,2]
            other = e["other_robot_"+task+"_impulse_Ns"][50:]
        result = score(task,q,v,touch,support,other,float(scene["execution_kg"]),initial,goal)
        for k in ("strict","safe_strict"):
            assert result[k] == (row[k]=="True"), (source,k)
        for k in ("final_error","settled_s","upright_s"):
            if result.get(k) is None:
                assert not row.get(k)
            else:
                np.testing.assert_allclose(result[k],float(row[k]),atol=1e-7)


def policy_pairs():
    """Case-paired B minus C at the same effective target, keeping signs and plants separate."""
    out = []

    def add(source, condition, rows, identity, treatment, left, right, metrics):
        arms = {a: {r[identity]: r for r in rows if r[treatment] == a} for a in [left, right]}
        assert arms[left].keys() == arms[right].keys()
        for metric in metrics:
            ids = sorted((case for case in arms[left] if all(arms[a][case][metric]
                         for a in [left, right])), key=int)
            differences = np.array([float(arms[left][case][metric])-float(arms[right][case][metric])
                                    for case in ids])
            low, high = interval(ids, differences)
            out.append(dict(source=source, condition=condition, left=left, right=right,
                            metric=metric, paired_cases=len(ids), lower_left=(differences < 0).sum(),
                            higher_left=(differences > 0).sum(), mean_left_minus_right=differences.mean(),
                            paired_low=low, paired_high=high))

    loads = [r for r in read("fixed_policy_load_trials") if r["mode"] in ("B", "C")]
    for r in loads:
        r["mass_stratum"] = r["mass_kg"] if r["task"] == "box" else "paired masses"
    for (task, mass, level), rows in grouped(loads, ["task", "mass_stratum", "level_N"]).items():
        metrics = (["progress_m", "lateral_m", "impulse_x_Ns", "ee_rms_m"] if task == "box"
                   else ["effective_v_rms", "unloaded_v_rms", "progress_m", "tension_peak_N"]
                   if task == "tether" else ["height_m", "load_height_shift_m", "nominal_z_rms_m"])
        add("fixed_policy_load_trials", f"{task} / {mass} / {level} N", rows, "case", "mode",
            "B", "C", metrics)
    for file in ["fixed_finger_pulse_trials", "final_pulse"]:
        rows = [r for r in read(file) if file != "final_pulse" or int(r["case"]) >= 4201]
        add(file, "held-out", rows, "case", "mode", "B", "C",
            ["peak_drop_m", "nominal_z_rms_m", "recovery_s"])
    add("tow_velocity_correction", "loaded", read("tow_velocity_correction"), "seed", "treatment",
        "B-16", "corrected", ["tracking_rms_m_s", "mean_speed_m_s", "progress_m"])
    add("supplement_wall", "held-out", [r for r in read("supplement_wall") if r["split"] == "held"],
        "case", "arm", "B", "C", ["mean_normal_N", "duty", "longest_loaded_s"])
    write("paired_policy_effects", out)


def main():
    OUT.mkdir(exist_ok=True)
    frozen = read("final_trials")
    scenes = {r["scene_seed"]:r for r in read("final_scenes")}
    assert len(frozen)==538 and len(scenes)==51
    assert Counter(r["block"] for r in frozen)==dict(general=282,challenge=144,sumo=112)
    lock = hashlib.sha256((DOC/"final_lock.json").read_bytes()).hexdigest()
    rows = []
    clocks = defaultdict(list)
    supplemental = read("supplement_runtime")
    for r in frozen + supplemental:
        sub = "block" not in r
        s = r["scene"] if sub else r["scene_seed"]
        if sub:
            path = Path("artifacts/supplemental/sumo/paired") / s / r["seed"]
            r.update(scene_seed=s,cem_seed=r["seed"],block="supplement_runtime",physics="known",
                     arm="runtime_full")
        else:
            path = Path("artifacts/final") / s / f"{r['cem_seed']}-{r['physics']}-{r['arm']}"
            assert json.loads((path/"final.json").read_text())["lock"]==lock
        verify(r,path,scenes[s],sub)
        keys = tuple(r[k] for k in ("block", "task", "physics", "arm"))
        with np.load(path / "episode.npz") as episode:
            if "planning_s" in episode:
                plan = episode["planning_s"]
            elif "latency_s" in episode:
                plan = episode["latency_s"]
            else:
                with np.load(path / "plans.npz") as plans:
                    plan = plans["planning_s"]
            assert len(plan) == int(r["updates"])
            np.testing.assert_allclose(np.percentile(plan, [50, 95, 100])*1000,
                                       [float(r[k]) for k in ("p50_ms", "p95_ms", "max_ms")])
            clocks[keys + ("plan",)].append(plan)
            if "cycle_s" in episode:
                clocks[keys + ("cycle",)].append(episode["cycle_s"])
            elif (path / "cycle_s.npy").exists():
                clocks[keys + ("cycle",)].append(np.load(path / "cycle_s.npy"))
        rows.append(r)
    groupkeys = ["block","task","physics","arm"]
    gs = grouped(rows,groupkeys)
    timing = []
    for key, chunks in clocks.items():
        ms = np.concatenate(chunks)*1000
        rs = gs[key[:-1]]
        period = float(rs[0]["period_ms"])
        timing.append(dict(zip(groupkeys, key[:-1]), timer=key[-1], episodes=len(chunks),
                           period_ms=period, misses_period=int(sum(ms > period)),
                           misses_100ms=int(sum(ms > 100)), **stats(ms)))
    write("pooled_timing_ms", timing)
    archives = grouped(read("final_groups"),groupkeys)
    metrics, failures, arms, pairs = [],[],[],[]
    for key, group in gs.items():
        desc = dict(zip(groupkeys,key))
        names = [r["scene_seed"] for r in group]
        success = np.array([r["strict"]=="True" for r in group])
        lo,hi = interval(names,success)
        err = np.array([float(r["final_error"]) for r in group])
        if key in archives:
            ref = archives[key][0]
            assert success.sum()==int(ref["success"])
            np.testing.assert_allclose([err.mean(),lo,hi],[float(ref[i]) for i in
                                       ("mean_error","success_ci_lo","success_ci_hi")])
        arms.append(dict(**desc,scenes=len(set(names)),episodes=len(group),strict=int(success.sum()),
                         rate=success.mean(),scene_ci_low=lo,scene_ci_high=hi,
                         error_unit="deg" if key[1]=="cone" else "m",mean_error=err.mean()))
        for k,n in Counter(r["status"] for r in group).items():
            failures.append(dict(**desc,status=k,count=n))
        for k in ("final_error","sim_s","settled_s","upright_s","contact_s","hand_impulse_Ns",
                  "hand_tick_peak_N","other_Ns","max_tilt_deg","valid_fraction","prefix_xy_rms_mm",
                  "prefix_hand_rms_N","overshoot_m","undershoot_m","cross_track_m","tension_mean_N",
                  "tension_peak_N","taut_entries","axial_m","tipping_Nms","translation_m"):
            values = [r[k] for r in group if r.get(k) not in (None,"")]
            if values:
                metrics.append(dict(**desc,metric=k,**stats(values)))
    for r in read("final_pairs"):
        left,right = [gs[tuple(r[k].split("/"))] for k in ("first","second")]
        ix = {(t["scene_seed"],t["cem_seed"]):t for t in right}
        p = [(t,ix[t["scene_seed"],t["cem_seed"]]) for t in left if (t["scene_seed"],t["cem_seed"]) in ix]
        a = np.array([x["strict"]=="True" for x,y in p])
        b = np.array([y["strict"]=="True" for x,y in p])
        assert (len(p),int(sum(a & ~b)),int(sum(b & ~a)))==tuple(int(r[k]) for k in
                ("pairs","first_only","second_only"))
        d = np.array([float(x["final_error"])-float(y["final_error"]) for x,y in p])
        np.testing.assert_allclose(d.mean(),float(r["error_delta"]),atol=1e-12)
        ids = [x["scene_seed"] for x, _ in p]
        el, eh = interval(ids, d)
        sl, sh = interval(ids, a.astype(int) - b.astype(int))
        np.testing.assert_allclose([el, eh], json.loads(r["error_delta_ci"]), atol=1e-12)
        np.testing.assert_allclose([sl, sh], json.loads(r["success_delta_ci"]), atol=1e-12)
        shared = [float(x["settled_s"])-float(y["settled_s"]) for x, y in p
                  if x["strict"] == y["strict"] == "True"]
        if shared:
            np.testing.assert_allclose(np.mean(shared), float(r["joint_time_delta_s"]), atol=1e-12)
        pairs.append(dict(first=r["first"], second=r["second"], scenes=len(set(ids)),
                          first_only=int(sum(a & ~b)), second_only=int(sum(b & ~a)),
                          paired_error_low=el, paired_error_high=eh, rate_low=sl, rate_high=sh,
                          jointly_solved=len(shared), joint_mean_time_delta_s=np.mean(shared) if shared else "",
                          **stats(d)))
    write("mpc_groups",arms)
    write("mpc_distributions",metrics)
    write("mpc_status",failures)
    write("paired_endpoint_differences",pairs)
    raw, contrasts = [],[]
    for name in ("supplement_push","supplement_push_heavy"):
        for cohort in ("replay","held","heavy_held"):
            subset = [r for r in read(name) if r["cohort"]==cohort]
            if not subset:
                continue
            by_arm = grouped(subset,["arm"])
            for (arm,),rs in by_arm.items():
                in_range, speed_pass, end = [],[],[]
                for r in rs:
                    with np.load(Path("artifacts/supplemental/push")/cohort/r["case"]/(arm+".npz")) as t:
                        pos = np.linalg.norm(t["qpos"][-25:,-7:-5]-t["goal_xy"],axis=1)
                        speed = np.linalg.norm(t["qvel"][-5:,-6:-4],axis=1)
                        np.testing.assert_allclose(pos[-1], float(r["final_error"]), atol=1e-10)
                        in_range.append(pos.max()<=.015)
                        speed_pass.append(speed.max()<=.02)
                        end.append(in_range[-1] and speed_pass[-1])
                        assert end[-1]==(r["end_settled"]=="True")
                for k in ("final_error","nominal_rms_mm","effective_rms_mm","contact_s","peak_speed",
                          "progress_m","impulse_x_Ns","abs_lateral_impulse_Ns","overshoot_m","undershoot_m",
                          "cross_track_m","mean_Fx_N","mean_Fy_N","peak_tick_accel"):
                    raw.append(dict(cohort=cohort,arm=arm,inside_final_25=sum(in_range),
                                    slow_final_5=sum(speed_pass),end_settled=sum(end),metric=k,
                                    **stats([r[k] for r in rs])))
            b = {r["case"]:r for r in by_arm[("B",)]}
            for (arm,),rs in by_arm.items():
                if arm=="B":
                    continue
                gains = [(b[r["case"]]["end_settled"]=="True")-(r["end_settled"]=="True") for r in rs]
                delta = [1000*(float(b[r["case"]]["final_error"])-float(r["final_error"])) for r in rs]
                ids = [r["case"] for r in rs]
                lo,hi = interval(ids,delta)
                sl,sh = interval(ids,gains)
                contrasts.append(dict(cohort=cohort,ref=arm,placements=len(ids),B_only=gains.count(1),
                                      ref_only=gains.count(-1),lower_B_error=sum(v<0 for v in delta),
                                      mean_B_minus_ref_mm=np.mean(delta),ci_low_mm=lo,ci_high_mm=hi,
                                      success_delta=np.mean(gains),ci_low_rate=sl,ci_high_rate=sh))
    write("push_distributions",raw)
    write("push_pairs",contrasts)
    # Descriptive summaries keep command sign, posture, physical mass and cohort identifiable.
    policy = []
    specifications = [
        ("fixed_policy_load_trials", ["task","mode","level_N","mass_group"],
         ["progress_m","lateral_m","effective_v_rms","unloaded_v_rms","time_25cm_s","post_taut_hold_s",
          "tension_mean_N","tension_peak_N","taut_duty","impulse_x_Ns","mean_Fx_N","peak_Fx_N",
          "contact_duty","ee_rms_m","height_m","load_height_shift_m","nominal_z_rms_m",
          "effective_z_rms_m","unloaded_posture_rms_m","arm_tau_rms_Nm","arm_clip_fraction","max_tilt_deg"]),
        ("fixed_finger_pulse_trials", ["mode"], ["peak_drop_m","recovery_s","nominal_z_rms_m","arm_peak_delta_rad"]),
        ("final_pulse", ["mode"], ["peak_drop_m","recovery_s","nominal_z_rms_m","arm_peak_delta_rad"]),
        ("tow_velocity_correction", ["treatment"], ["tracking_rms_m_s","progress_m","time_25cm_s"]),
        ("supplement_wall", ["arm"], ["mean_normal_N","duty","peak_normal_N","longest_loaded_s","effective_pre_rms_m"]),
        ("supplement_mismatch", ["task","plant","arm","informed"], ["final_error","prefix_xy_mm","prefix_ee_mm","prefix_force_N"]),
        ("wall_sliding_trials", ["variant","mode"], ["duty","rmse_N","traverse_rmse_N","hand_y_m","yaw_peak_deg"]),
        ("base_load_trials", ["split","controller","dimensions"], ["path_rmse_m","velocity_rmse_m_s","final_error_m"])]
    for name,fields,quantities in specifications:
        selected = []
        for r in read(name):
            if name=="final_pulse" and int(r["case"])<4201:
                continue
            if name.startswith("supplement") and r["split"]!="held":
                continue
            if name=="fixed_policy_load_trials":
                r["mass_group"] = r["mass_kg"] if r["task"]=="box" else "sampled"
            if name=="base_load_trials":
                r["split"] = "held" if r["case"].startswith("held") else "dev"
            selected.append(r)
        for key,group in grouped(selected,fields).items():
            for k in quantities:
                vals = [float(r[k]) for r in group if r.get(k) and np.isfinite(float(r[k]))]
                if vals:
                    policy.append(dict(source=name,group=" / ".join(key),metric=k,assigned=len(group),**stats(vals)))
    write("policy_and_secondary",policy)
    policy_pairs()
    extended = []
    for r in read("supplement_runtime_pairs"):
        t,ref = r["task"],r["reference"]
        left = gs[("supplement_runtime",t,"known","runtime_full")]
        right = gs[("sumo" if ref in ("native","work") else "general",t,"known",ref)]
        right = {(x["scene_seed"],x["cem_seed"]):x for x in right}
        ids,sd,ed,joint = [],[],[],[]
        for x in left:
            y = right[(x["scene_seed"],x["cem_seed"])]
            ids.append(x["scene_seed"])
            sd.append(int(x["strict"]=="True")-int(y["strict"]=="True"))
            ed.append(float(x["final_error"])-float(y["final_error"]))
            if x["strict"]==y["strict"]=="True":
                joint.append(float(x["settled_s"])-float(y["settled_s"]))
        assert sd.count(1)==int(r["runtime_only"]) and sd.count(-1)==int(r["reference_only"])
        np.testing.assert_allclose([np.mean(ed),np.mean(joint)],
                                   [float(r["mean_error_delta"]),float(r["mean_joint_delta_s"])])
        sl,sh = interval(ids,sd)
        el,eh = interval(ids,ed)
        extended.append(dict(**r,scenes=len(set(ids)),rate_delta=np.mean(sd),rate_ci_low=sl,
                             rate_ci_high=sh,error_ci_low=el,error_ci_high=eh))
    write("runtime_paired_intervals",extended)
    inventory = []
    for path in sorted(DOC.glob("*.csv")):
        records = read(path.stem)
        inventory.append(dict(file=path.name,rows=len(records),columns=len(records[0]),
                              sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    write("source_tables",inventory)
    print(f"Verified {len(frozen)} frozen, {len(supplemental)} supplemental ReLIC episodes, "
          f"{len(pairs)} frozen paired contrasts, 132 fixed-goal end holds.")


if __name__=="__main__":
    main()
