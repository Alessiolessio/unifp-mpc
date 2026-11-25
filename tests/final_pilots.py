"""Development-only SUMO fixture/observer pilots using previously executed UniFP scenes."""
import json
import os
from pathlib import Path
import subprocess

ROOT = Path("artifacts")


def relic(task, scene, out, seed, seconds, count, goal=None, parity=False, capture=False):
    cmd = ["../sumo/.pixi/envs/default/bin/python", "tests/sumo_common.py", task, str(scene),
           str(out), "--seed", str(seed), "--seconds", str(seconds), "--count", str(count)]
    if goal is not None:
        cmd.extend(["--goal", *map(str, goal)])
    if parity:
        cmd.append("--parity")
    subprocess.run(cmd, check=True, capture_output=capture, text=True, env=dict(os.environ, LD_PRELOAD=str(Path("build/sumo_observer.so").resolve()),
                   PYTHONDONTWRITEBYTECODE="1", PYTHONPATH="tools", OPENBLAS_NUM_THREADS="1"))


if __name__ == "__main__":
    for task, cases in [("box", ["00", "06", "08", "10"]), ("cone", ["near-0", "varied-305", "general-310", "general-82"])]:
        for i, case in enumerate(cases):
            folder = ROOT/("uncertainty/box" if task=="box" else "cone_force")/case
            src = folder/"nominal" if task=="box" else folder
            summaries = list(folder.glob("off/summary.json") if task=="box" else folder.glob("*-16-off/summary.json"))
            s = json.loads(sorted(summaries)[0].read_text())
            for count in ([0,6 if task=="box" else 8,16] if i==0 else [0]):
                out = ROOT/"final_dev_confirm"/task/case/str(count)
                if (out/"done").exists():
                    continue
                relic(task, src/"scene.mjb", out, s.get("seed",s.get("cem_seed")),
                      s["control_limit_s"], count, s.get("goal_xy_m"), i==0)
                (out/"done").touch()
