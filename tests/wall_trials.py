"""Run matched wall force trials serially so timings exclude competing experiments."""
import argparse
from functools import partial
import json
from pathlib import Path

from wall_mpc import POLICY, Rollouts, World, episode, spawn, wall_cost, wall_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", type=int, nargs="+")
    parser.add_argument("--out", type=Path, default=Path("artifacts/wall"))
    parser.add_argument("--horizon", type=float, choices=(.5, 1.), default=.5)
    parser.add_argument("--objective", choices=("tick", "impulse"), default="tick")
    parser.add_argument("--tangent-scale", type=float, default=.12)
    parser.add_argument("--padded", action="store_true")
    args = parser.parse_args()
    model = wall_model(args.padded)
    pool = None
    # First three stationary setpoints and the positive walk are development.
    cases = [(0, 17, 5, 0.), (0, 17, 3, 0.), (0, 17, 8, 0.), (0, 17, 5, .45),
             (42, 43, 6, 0.), (91, 67, 5, -.45), (0, 17, 15, 0.), (0, 17, 15, .45),
             (42, 43, 15, 0.), (91, 67, 15, -.45)]
    for case in args.cases:
        scene_seed, cem_seed, force, distance = cases[case]
        parent = args.out / str(case)
        parent.mkdir(parents=True, exist_ok=True)
        path = parent / "scene.mjb"
        x, yaw = spawn(model, path, scene_seed)
        for mode in ("off", "conditioned", "matched"):
            if pool is None:
                live = World(str(path), POLICY)
                pool = Rollouts(live, 8)
            else:
                live = pool.new_episode(str(path))
            folder = parent / mode
            result = episode(live, pool, force, distance, mode, cem_seed,
                             8. if distance else 5., folder=folder, horizon=args.horizon,
                             score=partial(wall_cost, objective=args.objective,
                                           tangent_scale=args.tangent_scale))
            result.update(scene=str(path), scene_seed=scene_seed, spawn_x_m=x, yaw_rad=yaw,
                          padded=args.padded, objective=args.objective, tangent_scale_m=args.tangent_scale)
            folder.mkdir(exist_ok=True)
            (folder/"summary.json").write_text(json.dumps(result, indent=2) + "\n")
            print(case, mode, result["status"], result.get("whole"), flush=True)


if __name__ == "__main__":
    main()
