"""Score physical outcomes and compare complete winning requests on the execution plant."""
import csv
import json
from pathlib import Path

import numpy as np

from cone_contact_wrench import analyze
from kinematics import nominal_targets
from model_uncertainty import POLICY, ROOT
from unifp_native import Rollouts, World


def write(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w") as f:
        writer = csv.DictWriter(f, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def angle(q, p):
    q, p = q/np.linalg.norm(q), p/np.linalg.norm(p)
    chord = min(np.linalg.norm(q-p), np.linalg.norm(q+p))
    return np.rad2deg(4*np.arcsin(min(1., chord/2)))


def metrics(folder):
    s = json.loads((folder/"summary.json").read_text())
    cone = s.get("task") == "cone"
    with np.load(folder/"episode.npz") as file:
        e = dict(file)
    force = "gripper_on_cone_N" if cone else "gripper_on_box_N"
    hand = np.linalg.norm(e[force][50:], axis=-1)
    keys = dict(task="cone" if cone else "box", case=int(folder.parent.name),
                scene=s["physical_seed"], cem=s["optimizer_seed"], mode=s["treatment"],
                informed=s.get("informed", False), execution_kg=s["execution_kg"],
                prediction_kg=s.get("prediction_kg", s.get("nominal_kg")), floor_scale=s["floor_scale"])
    row = dict(**keys, status=s["status"], control_s=(len(e["qpos"])-50)*.02,
               tilt_deg=s["max_abs_roll_pitch_deg"], hand_peak_N=hand.max(initial=0.),
               hand_impulse_Ns=.02*hand.sum(), contact_s=.02*e["contact_duty"][50:].sum(),
               issue_virtual_mean_N=np.linalg.norm(e["policy_command"][50:, 6:9], axis=1).mean(),
               other_Ns=s["other_robot_cone_impulse_Ns" if cone else "other_robot_box_impulse_Ns"],
               **s["planning"])
    u = e["forecast_request"][:, :5]
    xyz = u[..., :3].reshape(-1, 3)
    clipped = nominal_targets(e["qpos"][49:-1,:7], xyz) - xyz
    row["workspace_projected_fraction"] = np.mean(np.linalg.norm(clipped, axis=1) > 1e-5)
    if cone:
        row["valid_fraction"] = s["valid_candidates"] / (s["candidates"]*s["mpc_updates"])
        wrench = analyze(folder, trace=e, start=50)
        row.update(final_deg=s["final_angle_deg"], min_deg=s["min_angle_deg"],
                   upright_s=s["sumo_first_s"], released_s=s["control_s"] if s["status"]=="success" else None,
                   ground_final_N=s["final_ground_z_N"], hand_final_N=s["final_gripper_N"],
                   com_xy_m=np.linalg.norm(s["displacement_com_m"][:2]),
                   tipping_Nms=wrench["supported_tipping_Nms"], axial_m=wrench["impulse_weighted_axial_m"])
        lat = e["planning_s"]
    else:
        start, goal = np.array(s["box_start_xy_m"]), np.array(s["goal_xy_m"])
        delta = goal-start
        unit = delta/np.linalg.norm(delta)
        residual = e["qpos"][50:, -7:-5]-goal
        along = residual @ unit
        row.update(error_mm=1000*s["final_error_m"], settled_s=s["settle_times_s"]["precision"],
                   overshoot_mm=1000*max(0., along.max()), end_short_mm=1000*max(0., -along[-1]),
                   lateral_end_mm=1000*s["lateral_drift_m"],
                   peak_cross_mm=1000*abs(residual @ [-unit[1], unit[0]]).max(),
                   cycle_p95_ms=s["cycle"]["p95_ms"])
        with np.load(folder/"plans.npz") as p:
            row["valid_fraction"] = p["valid_candidates"].sum() / (s["candidates"]*s["mpc_updates"])
            lat = p["planning_s"]
    row["updates"] = len(lat)
    # Live state and the stored, nominal winner are aligned at every replan.
    # Branch on the execution plant, with identical future high-level requests.
    live = World(s["scene"], POLICY)
    oracle = Rollouts(live, 1)
    q = live.run(e["policy_command"][:50])["qpos"]
    np.testing.assert_array_equal(q, e["qpos"][:50])
    matched = s["treatment"].startswith("matched")
    branches = []
    for i, t in enumerate(e["forecast_start_tick"]):
        requests = e["forecast_request"][i]
        a = oracle.evaluate(live.snapshot(), requests[None], matched_position=matched)
        n = a["valid_steps"][0]
        if s["scene"] == s["prediction_scene"] or (
                keys["execution_kg"] == keys["prediction_kg"] and keys["floor_scale"] == 1.):
            for k in ("qpos", "qvel", "ee_world", "policy_command", force, "contact_duty"):
                np.testing.assert_array_equal(a[k][0], e["forecast_"+k][i])
        actual = live.run(requests[:5], matched_position=matched)
        for k in actual:
            np.testing.assert_array_equal(actual[k], e[k][t:t+5])
            np.testing.assert_array_equal(a[k][0,:5], actual[k])
        f = e["forecast_"+force][i]
        predicted = e["forecast_qpos"][i]
        contact = a["contact_duty"][0,:n] > 0
        nom = e["forecast_contact_duty"][i,:n] > 0
        first = np.flatnonzero(contact)
        nominal_first = np.flatnonzero(nom)
        for j in sorted({0, 4, 12, 24, len(requests)-1}):
            detail = dict(**keys, update=i, ms=20*(j+1), available=bool(n > j))
            if n > j:
                truth = a["qpos"][0,j]
                diff = predicted[j]-truth
                detail.update(base_mm=1000*np.linalg.norm(diff[:3]),
                              attitude_deg=angle(predicted[j,3:7],truth[3:7]),
                              finger_mm=1000*np.linalg.norm(e["forecast_ee_world"][i,j]-a["ee_world"][0,j]),
                              obj_xy_mm=1000*np.linalg.norm(diff[-7:-5]),
                              object_deg=angle(predicted[j,-4:],truth[-4:]),
                              obj_speed_err_m_s=np.linalg.norm(e["forecast_qvel"][i,j,-6:-3]-a["qvel"][0,j,-6:-3]),
                              force_vector_err_N=np.linalg.norm(f[j]-a[force][0,j]),
                              actual_contact=contact[j], predicted_contact=nom[j],
                              contact_disagree=nom[j]!=contact[j],
                              onset_diff_ms=20*(nominal_first[0]-first[0]) if len(first) and len(nominal_first) else None,
                              started_loaded=bool(e["contact_duty"][t-1] > 0),
                              horizon_contact_disagree=bool(len(first))!=bool(len(nominal_first)))
            branches.append(detail)
    return row, branches, lat


if __name__ == "__main__":
    rows, branches, latency = [], [], {"box": [], "cone": []}
    for task in latency:
        for path in sorted((ROOT/task).glob("*/*/summary.json")):
            row, diagnostic, lat = metrics(path.parent)
            rows.append(row)
            branches.extend(diagnostic)
            latency[task].extend(lat)
    write(Path("docs/uncertainty_trials.csv"), rows)
    write(ROOT/"branches.csv", branches)
    for task, times in latency.items():
        print(task, len(times), np.percentile(times, [50,95,100])*1000, np.sum(np.array(times)>.1))
    # Each horizon is a separate estimate; repeated replans are correlated.
    summary = []
    for task in latency:
        for mismatch in [False, True]:
            for ms in [20,100,260,500,1000]:
                group = [b for b in branches if b["ms"]==ms and b["task"]==task and
                         (not b["informed"] and (b["execution_kg"]!=b["prediction_kg"] or b["floor_scale"]!=1.))==mismatch]
                if not group:
                    continue
                valid = [b for b in group if b["available"]]
                loaded = [b for b in valid if b["actual_contact"] or b["predicted_contact"]]
                onsets = [abs(b["onset_diff_ms"]) for b in valid
                          if not b["started_loaded"] and b["onset_diff_ms"] is not None]
                summary.append(dict(new_onset_pairs=len(onsets), new_onset_p95_ms=np.percentile(onsets, 95),
                                    loaded=len(loaded),
                                    loaded_p95_force_N=np.percentile([b["force_vector_err_N"] for b in loaded],95),
                                    loaded_p95_xy_mm=np.percentile([b["obj_xy_mm"] for b in loaded],95),
                                    task=task, mismatch=mismatch, ms=ms, branches=len(group), valid=len(valid),
                                    **{k: np.mean([b[k] for b in valid]) for k in
                                    ["base_mm","attitude_deg","finger_mm","obj_xy_mm","object_deg",
                                     "obj_speed_err_m_s","force_vector_err_N","contact_disagree","horizon_contact_disagree"]},
                                    median_abs_onset_ms=np.median([abs(b["onset_diff_ms"]) for b in valid if b["onset_diff_ms"] is not None])))
    write(Path("docs/uncertainty_prediction.csv"), summary)
