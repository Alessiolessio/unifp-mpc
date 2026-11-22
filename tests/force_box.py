"""Paired full-base box episodes with the same physics, CEM seed and task cost."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from box_mpc import run
from box_scenario import generate

ROOT = Path("artifacts/force_box")
POLICY = Path("artifacts/encoder_actor.onnx")
# First two are development cases. Keep the correction case and second easy seed held out.
CASES = (("easy", 4, 17), ("general", 12, 17), ("general", 62, 43), ("easy", 17, 43),
         ("easy", 31, 67))
MODES = ("off", "conditioned", "matched")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mass", type=float, default=1.)
    parser.add_argument("--cases", nargs="+", type=int, default=range(4))
    args = parser.parse_args()
    for index in args.cases:
        tier, seed, optimizer = CASES[index]
        name = f"{tier}{seed}-{args.mass:g}kg"
        scene = ROOT / "scenes" / name
        generate(scene, seed, difficulty=tier, box_mass=args.mass)
        initial = None
        for mode in MODES:
            out = ROOT / name / mode
            options = SimpleNamespace(controller="mobile", seconds=None, policy=POLICY, warmup_ee=None,
                                      candidates=24, iterations=1, workers=8, seed=optimizer,
                                      out=out, ee_force=mode)
            s = run(options, scenario=scene, quiet=True)
            with np.load(out / "episode.npz") as episode:
                if initial is None:
                    initial = episode["qpos"][:50]
                np.testing.assert_array_equal(initial, episode["qpos"][:50])
            assert s["prefix_max_error"] == 0
            print(name, mode, s["status"], round(1000*s["final_error_m"], 2), s["planning"], flush=True)


if __name__ == "__main__":
    main()
