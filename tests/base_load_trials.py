"""Small matched loaded-locomotion study, run serially for comparable timings."""
import argparse
from pathlib import Path

import mujoco
import numpy as np

from base_load_mpc import World, Rollouts, episode

# Goal bearing and physical load in the fixed world, spawn yaw in degrees.
CASES = {
    "free": (0, 0, (0, 0), 17, False),
    "opp15": (0, 0, (-15, 0), 17, False),
    "opp25": (0, 0, (-25, 0), 17, False),
    "fast25": (0, 0, (-25, 0), 17, False),
    "side18": (0, 0, (0, 18), 17, False),
    "held_opp": (20, 35, (-19, -12), 43, False),
    "held_side": (-18, -35, (12, 17), 67, False),
    "held_free": (20, 35, (0, 0), 43, False),
    "held_pulse": (20, 35, (-8, 22), 43, True),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="+", choices=CASES)
    parser.add_argument("--two", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("artifacts/base_load"))
    args = parser.parse_args()
    scene = Path("artifacts/spot.mjb")
    pool = None
    for name in args.cases:
        yaw, angle, load, seed, pulse = CASES[name]
        # Store spawned scenes alongside traces so that recorded trajectories can be rendered.
        path = scene
        if yaw:
            path = args.out / f"spawn_{yaw}.mjb"
            path.parent.mkdir(parents=True, exist_ok=True)
            m = mujoco.MjModel.from_binary_path(str(scene))
            h = np.deg2rad(yaw) / 2
            m.key("spawn").qpos[3:7] = [np.cos(h), 0., 0., np.sin(h)]
            mujoco.mj_saveModel(m, str(path))
        delta = .9 * np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
        modes = ("conditioned", "matched") if args.two else ("off", "conditioned", "matched")
        for mode in modes:
            if pool is None:
                live = World(str(path), "artifacts/encoder_actor.onnx")
                pool = Rollouts(live, 8)
            else:
                live = pool.new_episode(str(path))
            r = episode(live, pool, mode, np.array(load), delta, seed, 5., pulse,
                        2 if args.two else 1, args.out/name/(mode+("2" if args.two else "")), path,
                        pace=.45 if name == "fast25" else .30)
            print(name, mode, r["success"], round(r["final_error_m"], 4),
                  round(r["path_rmse_m"], 4), r["hold_s"], flush=True)


if __name__ == "__main__":
    main()
