"""Compare the native drive to Isaac substep recordings, then drive a free MuJoCo robot."""

import argparse
import json
import runpy
import tempfile
from pathlib import Path

import mujoco
import numpy as np
from native_probe import DriveProbe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("../unifp-spot"))
    parser.add_argument("--scene", type=Path, default=Path("artifacts/spot.mjb"))
    args = parser.parse_args()
    model = mujoco.MjModel.from_binary_path(str(args.scene))
    order = np.argsort(model.actuator_user[:, 0])
    meta = model.actuator_user[order]
    q0 = meta[:, 1].astype(np.float32)
    archive = args.source / "outputs/validation/free_space_x_reset_verified"
    sim = json.loads((archive / "report.json").read_text())["simulation"]
    interface = sim["interface"]
    names = [model.joint(model.actuator_trnid[motor, 0]).name for motor in order]
    assert names == interface["joint_names_in_policy_order"]
    np.testing.assert_allclose(q0, interface["default_joint_position_rad"], atol=1e-7)
    trial = next(t for t in sim["trials"] if t["seed"] == 0 and t["case"]["name"] == "stand")
    for group in trial["actuators"].values():
        rows = [names.index(name) for name in group["joint_names"]]
        np.testing.assert_allclose(meta[rows, 2], group["stiffness"][0])
        np.testing.assert_allclose(meta[rows, 3], group["damping"][0])
        np.testing.assert_array_equal(meta[rows, 5], group["delay_steps"][0])
    replay, moving = DriveProbe(str(args.scene)), DriveProbe(str(args.scene))
    errors, angles = [], []
    recording = args.source / "outputs/validation/reset_verified/stand_0_reset.npz"
    with np.load(recording) as data:
        for i in range(1, 9):
            before = f"write_{i}_before/"
            target = data[before + "position_target_rad"][0]
            torque = replay.torque(data[before + "joint_position_rad"][0],
                                   data[before + "joint_velocity_rad_s"][0], target)
            reference = data[f"write_{i}_after/applied_effort_Nm"][0]
            errors.append(np.subtract(torque, reference))
            q = moving.step(target)
            angles.append(np.subtract(q, data[f"physics_{i}_after/joint_position_rad"][0]))
    print(f"8 x 19 recorded applied torques: max error {np.abs(errors).max():.6g} Nm")
    np.testing.assert_allclose(errors, 0, atol=2e-5)
    rms, worst = np.sqrt(np.mean(np.square(angles))), np.abs(angles).max()
    print("40 ms driven position error, rad: RMS", rms, "max", worst, "final [hip, knee, shoulder]",
          np.array(angles)[-1, [1, 11, 5]])
    assert rms < .01 and worst < .035, "Controlled reset response exceeds the transfer tolerance"

    # Continue with the archived policy targets in open loop. Score only samples after calibration.
    moving = DriveProbe(str(args.scene))
    with np.load(archive / "traces/seed_0_stand.npz") as data:
        angles = []
        for i, target in enumerate(data["joint_target_rad"][:10]):
            for _ in range(4):
                q = moving.step(target)
            if i >= 2:
                angles.append(q - data["joint_position_rad"][i + 1])
    rms, worst = np.sqrt(np.mean(np.square(angles))), np.abs(angles).max()
    print(f"60–200 ms driven response at policy ticks: RMS {rms:.6f}, max {worst:.6f} rad")
    assert rms < .02 and worst < .05

    constants = runpy.run_path(str(args.source / "unifp_spot/assets/spot/constants.py"))
    curve = np.asarray(constants["JOINT_PARAMETER_LOOKUP_TABLE"], dtype=np.float32)
    positions = np.r_[curve[0, 0]-1, curve[:, 0], (curve[:-1, 0]+curve[1:, 0])/2, curve[-1, 0]+1]
    worst = 0.0
    with tempfile.TemporaryDirectory() as tmp:
        for lag in (0, 3):
            model.actuator_user[:, 5] = lag
            path = str(Path(tmp) / "drive.mjb")
            mujoco.mj_saveModel(model, path)
            probe = DriveProbe(path)
            previous = [q0] * 4
            for t, angle in enumerate(positions):
                q = q0.copy()
                knee = meta[:, 4] == 0
                q[knee] = angle
                dq = np.full(19, 0.25, dtype=np.float32)
                target = q0 + (12 if t % 2 else -12)
                previous.append(target)
                actual = probe.torque(q, dq, target)
                limit = meta[:, 4].copy()
                limit[knee] = np.interp(angle, curve[:, 0], curve[:, 2])
                expected = np.clip(meta[:, 2]*(previous[-1-lag]-q) - meta[:, 3]*dq, -limit, limit)
                worst = max(worst, np.max(np.abs(actual-expected)))
    np.testing.assert_allclose(worst, 0, atol=1e-5)
    print(f"lags 0/3, signed saturation, knee knots, midpoints and endpoints: {worst:.6g} Nm")


if __name__ == "__main__":
    main()
