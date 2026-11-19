"""Replay the archived actor inputs and evaluate FK at the executed Isaac poses."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

import native_probe


def load_model(judo, menagerie_spot):
    spec = mujoco.MjSpec.from_file(str(judo / "judo/models/xml/spot_primitive/robot.xml"))
    for mesh in spec.meshes:
        mesh.file = str(menagerie_spot / "assets" / Path(mesh.file).name)
    for texture in spec.textures:
        if texture.file:
            texture.file = str(menagerie_spot / "spot.png")
    return spec.compile()


def check_free_space(path, interface, policy):
    with np.load(path) as trace:
        x, y, z, w = trace["root_quaternion_xyzw"].T
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
        default = np.asarray(interface["default_joint_position_rad"], dtype=np.float32)
        command = trace["command"]
        action = trace["action"].clip(-100, 100)
        previous = np.vstack((np.zeros_like(action[:1]), action[:-1]))
        phase = np.arange(len(action), dtype=np.float32) * 0.02 / 0.64 * 2 * np.pi
        frames = np.column_stack((
            roll, pitch, trace["angular_velocity_body_rad_s"] * 0.25,
            trace["joint_position_rad"] - default, trace["joint_velocity_rad_s"] * 0.05,
            previous, np.sin(phase), np.cos(phase),
            command[:, :2] * 2, command[:, 2] * 0.25, command[:, 3:6],
            np.zeros_like(command[:, 3:6]), command[:, 6:] * 0.01,
        )).clip(-100, 100)
        reference = trace["observation"]
        np.testing.assert_allclose(frames, reference, rtol=1e-5, atol=1e-5, equal_nan=False)
        np.testing.assert_allclose(default + 0.25 * action, trace["joint_target_rad"], atol=1e-6)
        # This recording starts at reset. The first inference includes the current frame once.
        padded = np.pad(frames, ((31, 0), (0, 0)))
        ticks = [0, 1, 31, len(frames) // 2, len(frames) - 1]
        histories = np.stack([padded[t:t + 32].ravel() for t in ticks])
        actual = native_probe.infer(str(policy), histories)
        np.testing.assert_allclose(actual, trace["action"][ticks], atol=1e-5, rtol=1e-5,
                                   equal_nan=False)
        print(f"{path.stem}, {len(frames)} reconstructed frames, "
              f"max |Δobs| {np.max(np.abs(frames - reference)):.3g}, "
              f"max |Δaction| at five ticks {np.max(np.abs(actual - trace['action'][ticks])):.3g}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("../unifp-spot"))
    parser.add_argument("--judo", type=Path, default=Path("../judo"))
    parser.add_argument("--policy", type=Path, default=Path("artifacts/encoder_actor.onnx"))
    parser.add_argument(
        "--menagerie-spot", type=Path,
        default=(Path.home() / ".cache/robot_descriptions/mujoco_menagerie/boston_dynamics_spot"),
    )
    args = parser.parse_args()
    outputs = args.source / "outputs"
    report = outputs / "validation/free_space_x_reset_verified/report.json"
    interface = json.loads(report.read_text())["simulation"]["interface"]
    for case in ("walk", "x_+20N_force"):
        check_free_space(report.parent / f"traces/seed_0_{case}.npz", interface, args.policy)
    model = load_model(args.judo, args.menagerie_spot.resolve())
    data = mujoco.MjData(model)
    names = interface["joint_names_in_policy_order"]
    qadr = [model.joint(name).qposadr.item() for name in names]
    assert len(set(qadr)) == model.nu == 19
    base = model.joint("base").qposadr.item()
    print(f"Compiled {model.nu} actuators, {model.body_mass.sum():.8f} kg, "
          f"dt {model.opt.timestep} s")
    for task in ("reach", "wall"):
        with np.load(outputs / f"mpc/latency_cpu_constant_writes_seed0/{task}_mpc.npz") as trace:
            frames = np.nan_to_num(trace["observation"]).clip(-100, 100)
            # The recorded trajectory starts after warmup. Frame 31 is the first full window.
            histories = np.stack([frames[i - 31:i + 1].ravel() for i in range(31, len(frames))])
            reference = trace["action"][31:]
            for batch in (1, len(histories)):
                actual = native_probe.infer(str(args.policy), histories[:batch])
                np.testing.assert_allclose(actual, reference[:batch], rtol=1e-5, atol=1e-5,
                                           equal_nan=False)
                error = np.max(np.abs(actual - reference[:batch]))
                print(f"{task}, {batch} histories, native vs executed max |Δaction| {error:.3g}")

            joints = trace["joint_position_rad"]
            relative = joints[:-1] - interface["default_joint_position_rad"]
            np.testing.assert_allclose(frames[1:, 5:24], relative, atol=2e-6)
            np.testing.assert_allclose(frames[1:, 43:62], trace["action"][:-1], atol=2e-6)
            errors = []
            for pose, q, ee in zip(trace["root_pose_w"], joints, trace["ee_m"], strict=True):
                data.qpos[base:base + 3] = pose[:3]
                data.qpos[base + 3:base + 7] = pose[[6, 3, 4, 5]]  # xyzw to wxyz
                data.qpos[qadr] = q
                mujoco.mj_kinematics(model, data)
                errors.append(np.linalg.norm(data.body("arm_link_fngr").xpos - ee))
            max_error = max(errors)
            print(f"{task}, {len(errors)} posed states, max EE-origin distance {max_error:.3g} m")
            np.testing.assert_array_less(max_error, 1e-4)


if __name__ == "__main__":
    main()
