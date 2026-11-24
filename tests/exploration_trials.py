"""Paired CEM restarts from a common, contact-free executed snapshot per cone scene."""
import argparse
from itertools import groupby
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cone_mpc import cone_model, cone_proposals, generate_cone, run_cone
from run_world import command_schedule
from unifp_native import Rollouts, World

ROOT = Path("artifacts/exploration")
MODES = ("off", "conditioned-z", "matched-z", "conditioned-xyz", "matched-xyz")
# General 82 is development. Additional starts selected from geometry alone.
SPLITS = {"principal": [(82, s) for s in range(101, 111)],
          "secondary": [(411, 117), (412, 117)], "attribution": [(82, 104)],
          "validation": [(418, 131), (419, 137), (421, 139)],
          "replication": [(443, 151), (444, 157), (445, 163)]}


def run(split, modes, sampler):
    model = cone_model()
    for scene_seed, seeds in groupby(SPLITS[split], key=lambda x: x[0]):
        scene = ROOT / str(scene_seed)
        scenario = generate_cone(scene, scene_seed, model, "general")
        live = World(str(scene/"scene.mjb"), "artifacts/encoder_actor.onnx")
        pool = Rollouts(live, 8)
        warm = live.run(command_schedule("push")[:50])
        snapshot = live.snapshot()
        for _, seed in seeds:
            for treatment in modes:
                folder = scene / f"{seed}-{treatment}-{sampler}"
                folder.mkdir(exist_ok=True)
                if (folder/"episode.npz").exists():
                    continue
                if not (folder/"scene.mjb").exists():
                    (folder/"scene.mjb").symlink_to("../scene.mjb")
                mode, _, axes = treatment.partition("-")
                args = SimpleNamespace(seed=seed, candidates=16, workers=8, iterations=1,
                                       seconds=12., horizon=1., sumo_sampling=True, out=folder,
                                       objective="upright_buffer", ee_force=mode,
                                       force_axes=axes or "xyz", broad_proposals=sampler == "broad",
                                       joint_proposals=sampler in ("joint", "sham"))
                live.restore(snapshot)
                if sampler == "sham":
                    # Consume the existing geometry proposal draws, then keep all local lanes.
                    def discard(q, planner):
                        cone_proposals(q, planner)
                        return None

                    with patch("cone_mpc.cone_proposals", discard):
                        s = run_cone(args, folder, scenario, live, pool, warm=warm)
                else:
                    s = run_cone(args, folder, scenario, live, pool, warm=warm)
                print(scene_seed, seed, treatment, sampler, s["status"], s["final_angle_deg"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("split", choices=SPLITS)
    p.add_argument("modes", nargs="+", choices=MODES)
    choice = p.add_mutually_exclusive_group()
    choice.add_argument("--broad", dest="sampler", action="store_const", const="broad")
    choice.add_argument("--joint", dest="sampler", action="store_const", const="joint")
    choice.add_argument("--sham", dest="sampler", action="store_const", const="sham")
    p.set_defaults(sampler="local")
    a = p.parse_args()
    run(a.split, a.modes, a.sampler)
