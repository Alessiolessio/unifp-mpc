"""Replay a failed saved cone episode and identify its last batch's rejected contacts."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from cone_mpc import run_cone
from unifp_native import Rollouts, World


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    folder = parser.parse_args().folder
    s = json.loads((folder / "summary.json").read_text())
    args = SimpleNamespace(**s, seconds=s["control_limit_s"], seed=s["cem_seed"], out=None,
                           horizon=s["horizon_s"])
    scene = str(folder / "scene.mjb")
    live = World(scene, "artifacts/encoder_actor.onnx")
    pool = Rollouts(live, args.workers)
    batch = {}

    def evaluate(snapshot, commands):
        result = pool.evaluate(snapshot, commands)
        batch.update(result)
        return result

    result = run_cone(args, folder, s["scenario"], live, SimpleNamespace(evaluate=evaluate))
    assert result["status"] == s["status"] == "no_valid_plan"
    assert result["final_angle_deg"] == s["final_angle_deg"]
    model = mujoco.MjModel.from_binary_path(scene)
    data = mujoco.MjData(model)
    obj = model.body("traffic_cone").id
    bad = batch["other_robot_cone_impulse_Ns"] > 0
    bodies = set()
    for poses, ticks in zip(batch["qpos"], bad):
        for pose in poses[ticks]:
            data.qpos[:] = pose
            mujoco.mj_forward(model, data)
            for contact in data.contact:
                a, b = model.geom_bodyid[contact.geom]
                if obj in (a, b):
                    other = b if a == obj else a
                    name = model.body(other).name
                    if other and name not in ("arm_link_fngr", "arm_link_wr1"):
                        bodies.add(name)
    print("Native-valid", batch["valid"].sum(), "with nongripper impulse", bad.any(axis=1).sum(),
          "tick-end contacts", sorted(bodies))
