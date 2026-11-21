"""Matched full episodes for contact recovery and the oblique approach demonstration."""

import argparse
from pathlib import Path
from types import SimpleNamespace

from box_mpc import BASE_CONTROLS, run
from box_scenario import generate

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("cases", nargs="+")
p.add_argument("--modes", nargs="+", choices=BASE_CONTROLS, default=["forward", "stance"])
p.add_argument("--objective", default="bounded")
p.add_argument("--seed", type=int, default=17)
p.add_argument("--out", type=Path, default=Path("artifacts/recovery"))
p.add_argument("--candidates", type=int, default=24)
p.add_argument("--iterations", type=int, default=1)
args = p.parse_args()
root = Path("artifacts/recovery")
for case in args.cases:
    if case.startswith("oblique"):
        scene = root / case
        if not (scene / "scene.mjb").exists():
            generate(scene, int(case[7:]), difficulty="oblique")
    elif case == "long14":
        scene = Path("artifacts/robust_box/scenes/14")
    else:
        scene = Path("artifacts/viewer_study") / case
    for mode in args.modes:
        folder = args.out / f"{case}_{mode}_{args.objective}_{args.seed}"
        settings = SimpleNamespace(out=folder, policy="artifacts/encoder_actor.onnx", warmup_ee=None,
                                   seconds=9., seed=args.seed, controller="mobile",
                                   candidates=args.candidates, workers=8, iterations=args.iterations)
        result = run(settings, scenario=scene, objective=args.objective, base_control=mode, quiet=True)
        print(case, mode, result["status"], result["final_error_m"], result["planning"], flush=True)
