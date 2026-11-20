"""Exact continuation at reset, in partial history, and through gripper/box contact."""

import numpy as np
from run_world import command_schedule, spherical
from unifp_native import Rollouts, World


def box_world():
    return World("artifacts/spot_box.mjb", "artifacts/encoder_actor.onnx")


def compare(candidate, actual, i):
    for field, values in actual.items():
        np.testing.assert_array_equal(candidate[field][i], values, err_msg=field)


def check(workers):
    cmd = command_schedule("push")
    horizon = 25
    reference = box_world()
    expected = reference.run(cmd[:135])
    live = box_world()
    pool = Rollouts(live, workers)
    done = 0
    for start in (0, 7, 93):
        live.run(cmd[done:start])
        snapshot = live.snapshot()
        a = cmd[start:start + horizon].copy()
        b = a.copy()
        b[:, 5] += .14
        b[:, 6] = -10
        oracle = {k: v[start:start + horizon] for k, v in expected.items()}
        batch = pool.evaluate(snapshot, np.stack([a, b, b, a]))
        assert batch["valid"].all() and (batch["valid_steps"] == horizon).all()
        assert np.max(np.abs(batch["action"][0] - batch["action"][1])) > .001
        compare(batch, oracle, 0)
        compare(batch, oracle, 3)
        swapped = pool.evaluate(snapshot, np.stack([b, a, a, b]))
        for field in expected:
            np.testing.assert_array_equal(batch[field], swapped[field][[1, 0, 3, 2]])
            np.testing.assert_array_equal(batch[field][1], batch[field][2])
        if start == 93:
            assert expected["contact_duty"][start - 1] > 0 and oracle["contact_duty"][0] > 0
            assert np.ptp(oracle["qpos"][:, -7]) > .001
        # Two candidate batches have run. The live continuation still matches the independent oracle.
        live_ticks = 3
        observed = live.run(a[:live_ticks])
        for field in oracle:
            np.testing.assert_array_equal(observed[field], oracle[field][:live_ticks])
        done = start + live_ticks

    # Restore after a different command, including the contact warmstart and previous policy action.
    live.run(b)
    live.restore(snapshot)
    restored = live.run(a)
    compare(batch, restored, 0)
    # Failure in one slot must leave its prefix usable and the next worker assignment clean.
    bad = a.copy()
    bad[4, 3] = np.nan
    result = pool.evaluate(snapshot, np.stack([bad, a]))
    assert result["valid"].tolist() == [False, True] and result["valid_steps"].tolist() == [4, horizon]
    assert "non-finite" in result["failure"][0]
    for k in oracle:
        np.testing.assert_array_equal(result[k][0, :4], restored[k][:4])
        assert np.isnan(result[k][0, 4:]).all()
    compare(result, restored, 1)
    # Take a new snapshot from the executed prefix and then reuse the older snapshot.
    new = live.snapshot()
    next_cmd = cmd[start + horizon:start + horizon + 5]
    next_batch = pool.evaluate(new, next_cmd[None])
    compare(next_batch, live.run(next_cmd), 0)
    compare(pool.evaluate(snapshot, a[None]), restored, 0)
    print(f"{workers} worker(s): exact reset, partial-history and contact continuation;"
          " A/B/B/A, reorder, live isolation, two snapshots and failed candidate passed.")


def check_world_targets():
    live = box_world()
    prefix = live.run(command_schedule("push")[:93])
    saved = live.snapshot()
    goals = np.array([[.81, -.05, .44], [-1, 2, 4], [-1, -2, -4],
                      [.34, -.04, .64], [.84, 0, .49]], dtype=np.float32)
    direct = live.run(goals)
    # Recover the source planner's yaw-frame commands independently at each pre-action pose.
    roots = np.vstack((prefix["qpos"][-1], direct["qpos"][:-1]))
    w, x, y, z = roots[:, 3:7].T
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    dx, dy, dz = (goals - roots[:, :3]).T
    local = np.stack((np.cos(yaw)*dx + np.sin(yaw)*dy, -np.sin(yaw)*dx + np.cos(yaw)*dy, dz), axis=1)
    reference = spherical(local - [.35, 0, .12]).clip(
        [.15, np.deg2rad(-40), -np.pi/2], [.55, np.deg2rad(50), np.pi/2])
    np.testing.assert_allclose(direct["policy_command"][:, 3:6], reference, atol=2e-5)
    projected = Rollouts(live, 4).evaluate(saved, np.stack([goals[::-1], goals]))
    compare(projected, direct, 1)
    live.restore(saved)
    replay = live.run(direct["policy_command"])
    compare(projected, replay, 1)
    assert (replay["policy_command"][:, :3] == 0).all() and (replay["policy_command"][:, 6:] == 0).all()
    print("World targets: per-pose yaw conversion, workspace bounds and raw-command replay passed.")
    mobile = np.column_stack((goals, np.linspace(-.12, .24, len(goals), dtype=np.float32)))
    zero = mobile.copy()
    zero[:, 3] = 0
    batch = Rollouts(live, 2).evaluate(saved, np.stack([mobile, zero, mobile]))
    compare(batch, direct, 1)
    live.restore(saved)
    moving = live.run(mobile)
    compare(batch, moving, 0)
    compare(batch, moving, 2)
    np.testing.assert_array_equal(moving["policy_command"][:, 0], mobile[:, 3])
    assert not np.any(moving["policy_command"][:, 1:3]) and not np.any(moving["policy_command"][:, 6:])
    live.restore(saved)
    compare(batch, live.run(moving["policy_command"]), 0)
    print("World XYZ + body vx matched live execution, raw replay and the zero-speed baseline.")
    planar = np.column_stack((mobile, np.linspace(.16, -.12, len(goals), dtype=np.float32)))
    zero_y = planar.copy()
    zero_y[:, 4] = 0
    batch = Rollouts(live, 2).evaluate(saved, np.stack([planar, zero_y]))
    compare(batch, moving, 1)
    live.restore(saved)
    moving = live.run(planar)
    compare(batch, moving, 0)
    np.testing.assert_array_equal(moving["policy_command"][:, :2], planar[:, 3:5])
    live.restore(saved)
    compare(batch, live.run(moving["policy_command"]), 0)
    print("World XYZ + body vx/vy matched live execution, raw replay and zero lateral velocity.")
    # A sustained turn changes the world-to-shoulder transform during the forecast.
    turn = np.tile(np.array([.84, .1, .52, .12, .04, .45], np.float32), (40, 1))
    zero_turn = turn.copy()
    zero_turn[:, 5] = 0
    batch = Rollouts(live, 2).evaluate(saved, np.stack([zero_turn, turn]))
    live.restore(saved)
    compare(batch, live.run(turn[:, :5]), 0)
    live.restore(saved)
    turning = live.run(turn)
    compare(batch, turning, 1)
    raw = turning["policy_command"]
    np.testing.assert_array_equal(raw[:, :3], turn[:, 3:])
    assert not raw[:, 6:].any()
    roots = np.vstack((prefix["qpos"][-1], turning["qpos"][:-1]))
    w, x, y, z = roots[:, 3:7].T
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    assert np.ptp(np.unwrap(yaw)) > .1
    dx, dy, dz = (turn[:, :3] - roots[:, :3]).T
    local = np.column_stack((np.cos(yaw)*dx + np.sin(yaw)*dy, -np.sin(yaw)*dx + np.cos(yaw)*dy, dz))
    expected = spherical(local - [.35, 0, .12]).clip(
        [.15, np.deg2rad(-40), -np.pi/2], [.55, np.deg2rad(50), np.pi/2])
    np.testing.assert_allclose(raw[:, 3:6], expected, atol=2e-5)
    live.restore(saved)
    compare(batch, live.run(raw), 1)
    print("Turning world targets matched raw replay, independent frame conversion and zero-yaw control.")


if __name__ == "__main__":
    check(1)
    check(4)
    check_world_targets()
