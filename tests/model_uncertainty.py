"""Paired physics-mismatch pilots. Run serially for uncontended latency measurements."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from box_mpc import run
from box_scenario import generate
from cone_mpc import cone_model, generate_cone, run_cone
from unifp_native import Rollouts, World

ROOT = Path("artifacts/uncertainty")
POLICY = "artifacts/encoder_actor.onnx"
# tier, physical seed, CEM seed, nominal kg, execution mass/floor multipliers
BOX = [
    ("easy", 4, 17, 1, m, f)
    for m, f in [(1, 1), (.7, 1), (1.7, 1), (1, .65), (1, 1.4)]
] + [("easy", 4, 17, 3, 5/3, 1)] + [
    (tier, seed, cem, 1, m, f)
    for tier, seed, cem in [("easy", 1201, 211), ("medium", 1202, 223), ("oblique", 1203, 227)]
    for m, f in [(1, 1), (1.7, 1.4)]
] + [("easy", seed, cem, 3, 5/3, 1.) for seed, cem in [(1221, 233), (1222, 239)]]
# Historical winning and losing Z seeds, plus two fresh physical starts.
CONE = [(tier, seed, cem, scale)
        for tier, seed, cem in [("general", 82, 101), ("general", 82, 102),
                                ("varied", 1211, 211), ("general", 1212, 223)]
        for scale in ([1.25] if seed == 82 else [1., 1.25])] + [
            ("general", 82, 241, scale) for scale in [1., 1.25]]


def run_case(task, index, informed=False):
    cone = task == "cone"
    tier, seed, cem, *physics = (CONE if cone else BOX)[index]
    mass, scale, floor = (3.36, physics[0], 1.) if cone else physics
    dest = ROOT/task/f"{index:02d}"
    nominal = dest/"nominal"
    nominal.mkdir(parents=True, exist_ok=True)
    if cone:
        settings = generate_cone(nominal, seed, cone_model(), tier)
    else:
        settings = generate(nominal, seed, difficulty=tier, box_mass=mass)
    (nominal/"scenario.json").write_text(json.dumps(settings, indent=2))
    physical = dest/"execution.mjb"
    prediction = physical if informed else nominal/"scene.mjb"
    pool = Rollouts(World(str(prediction), POLICY), 8)
    modes = ["off", "conditioned", "matched"]
    if cone and seed == 1212:
        modes += ["conditioned-xyz", "matched-xyz"]
    for label in modes:
        out = dest/(label + ("-informed" if informed else ""))
        out.mkdir(exist_ok=True)
        live = pool.execution_world() if informed else pool.execution_world(scale, floor)
        # Store the actual scene for exact offline contact and forecast diagnostics.
        if label == "off" and not informed:
            live.save_model(str(physical))
        if not (out/"scene.mjb").exists():
            (out/"scene.mjb").symlink_to("../execution.mjb")
        mode = label.split("-")[0]
        args = SimpleNamespace(seed=cem, workers=8, iterations=1, policy=POLICY, out=out,
                               ee_force=mode, force_axes="xyz" if "-" in label else "z",
                               seconds=12. if cone else None, controller="mobile", warmup_ee=None,
                               candidates=16 if cone else 24, objective="upright_buffer",
                               sumo_sampling=cone, horizon=1. if cone else .5)
        if cone:
            summary = run_cone(args, out, settings, live, pool)
        else:
            summary = run(args, scenario=nominal, live=live, pool=pool, quiet=True)
        summary.update(scene=str(physical), prediction_scene=str(prediction), informed=informed,
                       prediction_kg=live.object_mass if informed else mass, execution_kg=live.object_mass, floor_scale=floor,
                       treatment=label, physical_seed=seed, optimizer_seed=cem)
        (out/"summary.json").write_text(json.dumps(summary, indent=2))
        with np.load(out/"episode.npz") as trace:
            if label == "off":
                warm = {k: trace[k][:50] for k in ("qpos", "qvel", "action", "policy_command")}
            else:
                for k, v in warm.items():
                    np.testing.assert_array_equal(v, trace[k][:50])
            if informed:
                with np.load(dest/label/"episode.npz") as blind:
                    np.testing.assert_array_equal(trace["qpos"][:50], blind["qpos"][:50])
            if not cone and index == 0:
                archive = Path("artifacts/force_box/easy4-1kg")/label/"episode.npz"
                if archive.exists():
                    with np.load(archive) as old:
                        for k in old.files:
                            if k != "cycle_s" and not k.startswith("forecast_"):
                                np.testing.assert_array_equal(trace[k], old[k], err_msg=k)
        print(task, index, tier, seed, label, summary["status"],
              summary.get("final_error_m", summary.get("final_angle_deg")), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", choices=["box", "cone"])
    p.add_argument("indices", nargs="*", type=int)
    p.add_argument("--informed", action="store_true", help="Replay with known execution physics")
    args = p.parse_args()
    for i in args.indices or range(len(BOX if args.task == "box" else CONE)):
        run_case(args.task, i, args.informed)
