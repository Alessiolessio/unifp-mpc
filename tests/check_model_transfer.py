"""Exact two-model continuation, including loaded contact and model identity guards."""
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np
from check_branching import box_world, compare
from run_world import command_schedule
from unifp_native import Rollouts, World


def check_layout():
    from tether_mpc import prepare, scene_template
    with TemporaryDirectory() as tmp:
        name = Path(tmp)/"tether.mjb"
        prepare(scene_template(tether=True), name, "straight")
        original = mujoco.MjModel.from_binary_path(str(name))
        nominal = World(str(name), "artifacts/encoder_actor.onnx")
        pool = Rollouts(nominal, 2)
        live = pool.execution_world(1.7, .65)
        live.save_model(str(name))
        actual = mujoco.MjModel.from_binary_path(str(name))
        for k in ("names", "nq", "nv", "na", "nu", "nbody", "njnt", "ngeom", "nsite",
                  "ntendon", "body_parentid", "body_pos", "body_quat", "jnt_bodyid", "jnt_axis",
                  "jnt_type", "jnt_qposadr", "jnt_dofadr", "geom_type", "geom_bodyid", "geom_size",
                  "actuator_trnid", "actuator_user", "actuator_gear", "numeric_data", "site_bodyid",
                  "tendon_adr", "tendon_num", "wrap_type", "wrap_objid", "tendon_range", "key_qpos"):
            np.testing.assert_array_equal(getattr(original, k), getattr(actual, k), err_msg=k)
        pair = original.pair("box_ground").id
        body = original.body("box_body").id
        for k in ("body_mass", "body_inertia", "pair_friction"):
            expected = getattr(original, k).copy()
            if k == "pair_friction":
                expected[pair, :2] *= .65
            else:
                expected[body] *= 1.7
            np.testing.assert_array_equal(expected, getattr(actual, k))
        sig = mujoco.mjtState.mjSTATE_INTEGRATION
        assert mujoco.mj_stateSize(actual, sig) == mujoco.mj_stateSize(original, sig)
        assert live.object_mass == 1.7 * nominal.object_mass


def main():
    check_layout()
    nominal = box_world()
    pool = Rollouts(nominal, 4)
    live = pool.execution_world(known_load=True)
    assert not pool.mismatched
    cmd = command_schedule("push")
    for i, j in ((0, 7), (7, 93), (93, 118)):
        if i == 93:
            nominal.set_torso_load(-4., 2.)
            live.set_torso_load(-4., 2.)
        snap = live.snapshot()
        alternatives = np.stack([cmd[i:j], cmd[i:j].copy()])
        alternatives[1, :, 0] += .1
        predicted = pool.evaluate(snap, alternatives)
        reference = pool.evaluate(nominal.snapshot(), alternatives)
        actual = live.run(cmd[i:j])
        compare(predicted, actual, 0)
        for key, value in nominal.run(cmd[i:j]).items():
            np.testing.assert_array_equal(value, actual[key])
            np.testing.assert_array_equal(predicted[key], reference[key])
        if i == 93:
            assert actual["contact_duty"].max() > 0
    other = pool.execution_world(mass_scale=2.)
    assert pool.mismatched
    for stale in (snap, box_world().snapshot()):
        try:
            pool.evaluate(stale, alternatives)
        except ValueError:
            pass
        else:
            raise AssertionError("foreign or superseded execution model accepted")
    other.run(cmd[:93])
    snap = other.snapshot()
    a = pool.evaluate(snap, alternatives)
    other.set_torso_load(25., 10.)
    b = pool.evaluate(other.snapshot(), alternatives[::-1].copy())
    for key in actual:
        np.testing.assert_array_equal(a[key], b[key][::-1])
    other.restore(snap)
    observed = other.run(alternatives[0])
    assert np.linalg.norm(observed['qpos'][-1,-7:-5]-a['qpos'][0,-1,-7:-5]) > .0001
    print("Separate identical models: every trace field exact at reset, tick 7 and loaded contact tick 93. Foreign models rejected. Hidden load, order and nonzero physical mismatch checked.")


if __name__ == "__main__":
    main()
