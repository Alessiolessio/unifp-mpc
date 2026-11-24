"""Six broad rate proposals at unchanged budget on three fixed steep-bearing goals."""
from pathlib import Path
import numpy as np

from prepare_scene import scene_template
from tether_mpc import World, Rollouts, prepare, episode

root = Path("artifacts/exploration_tether")
scene = root / "scene.mjb"
_, limit = prepare(scene_template(tether=True), scene, "side")
for bearing, seed in [(65, 67), (55, 149), (72, 151)]:
    delta = .8 * np.array([np.cos(np.deg2rad(bearing)),np.sin(np.deg2rad(bearing))])
    for broad in (False, True):
        live = World(str(scene), "artifacts/encoder_actor.onnx")
        folder = root / f"{bearing}-{'broad' if broad else 'local'}"
        s = episode(live, Rollouts(live, 8), "off", delta, limit, seed, 9., folder, scene,
                    "side", broad=broad)
        print(folder,s["status"],s["final_error_m"],flush=True)
