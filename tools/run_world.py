"""Run a scripted UniFP policy in a headless, single-world MuJoCo simulation."""

import argparse
from pathlib import Path

import numpy as np
from unifp_native import World


def spherical(xyz):
    xyz = np.asarray(xyz, dtype=np.float32)
    radius = np.linalg.norm(xyz, axis=-1)
    elevation = np.arcsin(np.clip(xyz[..., 2] / (radius + 1e-8), -1, 1))
    return np.stack((radius, elevation, np.arctan2(xyz[..., 1], xyz[..., 0])), axis=-1)


def command_schedule(task):
    box = task == "push"
    time = np.arange(500 if box else 300) * .02
    command = np.zeros((len(time), 12), dtype=np.float32)
    fraction = np.clip(time - 2, 0, 1)
    xyz = np.tile([.35, 0, .10], (len(time), 1))
    if box:
        xyz[:] = [.20, 0, .10]
        xyz[time >= 1] = [.48, 0, -.20]
    elif task == "walk":
        command[:, 0] = .3 * fraction
    elif task == "reach":
        xyz += fraction[:, None] * [.05, .10, .05]
    command[:, 3:6] = spherical(xyz)
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("stand", "walk", "reach", "push"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--save", type=Path, help="Save the executed 50 Hz state and contact trace")
    args = parser.parse_args()
    model = "spot_box.mjb" if args.task == "push" else "spot.mjb"
    world = World(str(args.artifacts / model), str(args.artifacts / "encoder_actor.onnx"))
    commands = command_schedule(args.task)
    trace = world.run(commands)
    elapsed = .02 * len(commands)
    print(f"{args.task}: completed {elapsed:g} simulated seconds")
    print("Final base position (m)", trace["qpos"][-1, :3])
    print("Final finger origin (m)", trace["ee_world"][-1])
    if args.task == "push":
        # The first second is a contact-free warmup. The box is the last free joint.
        delta = trace["qpos"][-1, -7:-4] - trace["qpos"][49, -7:-4]
        print("Box displacement after warmup (m)", delta)
        print(f"Peak forward gripper force: {trace['peak_forward_N'].max():.2f} N")
    if args.save:
        np.savez_compressed(args.save, command=commands, time_s=(np.arange(len(commands)) + 1) * .02,
                            **trace)


if __name__ == "__main__":
    main()
