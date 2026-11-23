"""Independently check applied impulse, load restoration, and reachable command equivalence."""
from pathlib import Path
import tempfile

import mujoco
import numpy as np

from base_load_mpc import KD, SPEED, effective_velocity, policy_commands, sample_rates
from cem import DT, TIMES, interpolate
from run_world import command_schedule
from unifp_native import Rollouts, World


def check_impulse():
    model = mujoco.MjModel.from_binary_path("artifacts/spot.mjb")
    model.key("spawn").qpos[2] += 1.
    cmd = command_schedule("stand")[:1]
    data = mujoco.MjData(model)
    momentum = []
    with tempfile.TemporaryDirectory() as tmp:
        scene = str(Path(tmp) / "airborne.mjb")
        mujoco.mj_saveModel(model, scene)
        world = World(scene, "artifacts/encoder_actor.onnx")
        pool = Rollouts(world, 2)
        initial = world.snapshot()
        load = np.array([-19., 12.])
        for force in (np.zeros(2), load):
            world.restore(initial)
            world.set_torso_load(*force)
            snapshot = world.snapshot()
            forecast = pool.evaluate(snapshot, np.stack((cmd, cmd)))
            world.set_torso_load(5., 6.)
            world.restore(snapshot)
            tick = world.run(cmd)
            np.testing.assert_array_equal(forecast["qpos"][:, 0], np.repeat(tick["qpos"], 2, 0))
            data.qpos[:] = tick["qpos"][0]
            data.qvel[:] = tick["qvel"][0]
            mujoco.mj_forward(model, data)
            mujoco.mj_subtreeVel(model, data)
            assert data.ncon == 0
            momentum.append(data.subtree_linvel[0, :2].copy() * model.body_subtreemass[0])
    measured = (momentum[1] - momentum[0]) / DT
    print("World XY force from free-flight whole-system momentum change (N)", measured)
    np.testing.assert_allclose(measured, load, atol=.2)


def check_controls():
    cmd = np.zeros((4, 12), np.float32)
    cmd[:, 9] = [20., 20.01, 6., 0.]
    cmd[2, 2] = .21
    np.testing.assert_allclose(effective_velocity(cmd, np.zeros(2))[:, 0],
                               [0., 20.01/KD, .03, 0.], atol=1e-7)
    rng = np.random.default_rng(10)
    force_rngs = [np.random.default_rng(i) for i in (11, 12)]
    basis = np.array([[.6, .8], [-.8, .6]])
    scale = np.r_[SPEED, 30., 30.]
    previous = np.zeros(5)
    mean = np.zeros((3, 5))
    for _ in range(60):
        knots = sample_rates(mean, np.ones_like(mean), mean, previous, scale, basis,
                             rng, force_rngs, 24)
        rates = interpolate(knots, TIMES) * scale
        b = policy_commands(rates, basis, "conditioned")
        c = policy_commands(rates, basis, "matched")
        ext = rng.uniform(-25, 25, (24, 25, 2)).astype(np.float32)
        np.testing.assert_allclose(effective_velocity(b, ext), effective_velocity(c, ext), atol=1e-7)
        for v in (b, c):
            assert np.all(abs(v[..., :3]) <= SPEED+1e-7)
            assert np.max(abs(np.diff(v[..., :3], axis=1))) <= .8*DT+1e-7
            assert np.all(v[..., 6:9] == 0) and np.all(v[..., 11] == 0)
        assert np.linalg.norm(b[..., 9:11], axis=-1).max() <= 30.00001
        assert np.linalg.norm(np.diff(b[..., 9:11], axis=1), axis=-1).max() <= 80*DT+1e-5
        previous = rates[2, 4]/scale
        mean = knots[2]
    print("Deadband, virtual and effective bounds, force/velocity slew and 2D target parity passed.")


if __name__ == "__main__":
    check_impulse()
    check_controls()
