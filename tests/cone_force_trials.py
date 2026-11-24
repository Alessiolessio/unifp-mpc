"""Matched clearance/release CEM comparisons. Save each scene once and each executed prefix."""
import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cone_mpc import angle_deg, cone_model, generate_cone, run_cone
from measure_force import nominal_targets
from box_mpc import world_ee_target
from unifp_native import Rollouts, World

ROOT = Path("artifacts/cone_force")
# Disjoint from all archived cone development and evaluation seeds.
HELD = [(tier, seed, (71, 79, 83, 89)[i % 4])
        for i, (tier, seed) in enumerate(zip(["near"]*4 + ["varied"]*4 + ["general"]*4,
                                              range(301, 313), strict=True))]
ORIGINAL = [("near", i, 17 if i < 4 else 29) for i in range(7)]
DIFFICULT = [("general", 82, 43), ("general", 82, 67), ("near", 1, 67)]


def trial(model, pool, case, treatment, axis, count):
    tier, seed, cem = case
    folder = ROOT/f"{tier}-{seed}"
    scenario = generate_cone(folder, seed, model, tier) if pool is None else None
    if scenario is not None:
        (folder/"spawn.json").write_text(json.dumps(scenario))
    else:
        scenario = json.loads((folder/"spawn.json").read_text())
    scene = str(folder/"scene.mjb")
    run = folder/f"{cem}-{count}-{treatment}"
    run.mkdir(exist_ok=True)
    if not (run/"scene.mjb").exists():
        (run/"scene.mjb").symlink_to("../scene.mjb")
    if pool is None:
        live = World(scene, "artifacts/encoder_actor.onnx")
        pool = Rollouts(live, 8)
    else:
        live = pool.new_episode(scene)
    mode = "off" if treatment == "off" else treatment.split("-")[0]
    axes = "xyz" if treatment.endswith("3") else axis
    args = SimpleNamespace(seed=cem, workers=8, candidates=count, iterations=1, seconds=12.,
                           horizon=1., objective="upright_buffer", out=run, sumo_sampling=True,
                           ee_force=mode, force_axes=axes)
    s = run_cone(args, run, scenario, live, pool)
    return pool, s


def metrics(path):
    s = json.loads(path.read_text())
    sc = s["scenario"]
    with np.load(path.parent/"episode.npz") as e:
        lat = e["planning_s"]*1000
        q = e["qpos"][49:]
        dt = .02
        raw = e["requests"]
        t = e["policy_command"][50:]
        hand = np.linalg.norm(e["gripper_on_cone_N"][50:], axis=1)
        angle = angle_deg(q[:, -4:])
        improvement = -np.diff(angle)
        cmd = t[:, 6:9]
        assert not t[:, 9:].any()
        virtual = np.linalg.norm(cmd, axis=1)
        want = np.zeros(len(t))
        shortened = np.zeros(len(t), bool)
        if s.get("ee_force", "off") != "off" and len(t):
            u = raw.reshape(-1, 9)
            assert np.abs(u[:,6:]).max() <= 40.00001
            assert np.abs(np.diff(u[:,6:],axis=0)).max(initial=0) <= 1.60002
            assert np.max(abs(raw[1:,0,6:]-raw[:-1,-1,6:]), initial=0) == 0
            want = np.linalg.norm(u[:,6:],axis=1)
            if s["ee_force"] == "matched":
                assert not cmd.any()
                nominal = nominal_targets(q[:-1,:7], u[:,:3])
                displaced = world_ee_target(q[:-1,:7], t[:,3:6])
                offset_N = 200*np.linalg.norm(displaced-nominal,axis=1)
            else:
                offset_N = virtual
            shortened = want-offset_N > .01
        return dict(tier=sc["difficulty"], seed=sc["scenario_seed"], cem=s["cem_seed"],
                    population=s["candidates"], treatment=path.parent.name.split("-",2)[2],
                    axis=s.get("force_axes",""), status=s["status"], final_deg=s["final_angle_deg"],
                    upright_s=s["sumo_first_s"], end_s=s["control_s"], held_s=s["settled_s"],
                    tilt_robot_deg=s["max_abs_roll_pitch_deg"], min_deg=s["min_angle_deg"],
                    xy_com_m=np.linalg.norm(s["displacement_com_m"][:2]),
                    contact_s=e["contact_duty"][50:].sum()*dt, peak_tick_N=hand.max(initial=0),
                    resultant_impulse_Ns=hand.sum()*dt,
                    contact_net_tilt_gain_deg=improvement[hand>1].sum(),
                    final_angular_rad_s=s["final_angular_speed_rad_s"],
                    final_support_N=s["final_ground_z_N"], final_hand_N=s["final_gripper_N"],
                    other_Ns=s["other_robot_cone_impulse_Ns"], requested_mean_N=want.mean(),
                    actor_mean_N=virtual.mean(), shortened_fraction=shortened.mean(),
                    valid_fraction=s["valid_candidates"]/(s["mpc_updates"]*s["candidates"]),
                    updates=len(lat), **s["planning"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("split", choices=("original","difficult","held","budget"))
    p.add_argument("axis", choices=("x","y","z"))
    p.add_argument("--count", type=int, default=16)
    args = p.parse_args()
    model = cone_model()
    cases = {"original": ORIGINAL, "difficult": DIFFICULT, "held": HELD,
             "budget": [("general", 82, 43), ("near", 1, 17)]}[args.split]
    for case in cases:
        pool = None
        modes = ("off", "conditioned-1", "matched-1", "conditioned-3", "matched-3")
        for mode in modes:
            pool, result = trial(model, pool, case, mode, args.axis, args.count)
            print(case, mode, result["status"], round(result["final_angle_deg"],2), flush=True)
    rows = [metrics(p) for p in sorted(ROOT.glob("*/*/summary.json"))]
    with (ROOT/"episodes.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=rows[0], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
