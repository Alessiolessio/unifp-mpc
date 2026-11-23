"""Matched closed-loop tow trials. Run sequentially to time full planning updates."""
import argparse
from pathlib import Path

from tether_mpc import CASES, Rollouts, World, episode, prepare, scene_template

SEEDS = {"straight": 17, "diagonal": 17, "heavy": 17, "reverse_diagonal": 43, "side": 67}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cases", choices=CASES, nargs="+")
    p.add_argument("--without-guidance", action="store_true", help="Ordinary-velocity ablation")
    p.add_argument("--horizon", choices=(.5, 1.), type=float, default=.5)
    args = p.parse_args()
    root = Path("artifacts/tow" if args.horizon == .5 else "artifacts/tow_1s")
    model = scene_template(tether=True)
    pool = None
    for case in args.cases:
        scene = root/case/"scene.mjb"
        delta, limit = prepare(model, scene, case)
        modes = ["off"] if args.without_guidance else ["off", "conditioned", "matched"]
        for mode in modes:
            if pool is None:
                live = World(str(scene), "artifacts/encoder_actor.onnx")
                pool = Rollouts(live, 8)
            else:
                live = pool.new_episode(str(scene))
            folder = root/case/("unguided" if args.without_guidance else mode)
            r = episode(live, pool, mode, delta, limit, SEEDS[case], 9., folder, scene, case,
                        guide=not args.without_guidance, horizon=args.horizon)
            print(case, folder.name, r["status"], round(r["final_error_m"], 4), r["simulated_s"],
                  round(r["p95_ms"], 2), flush=True)


if __name__ == "__main__":
    main()
