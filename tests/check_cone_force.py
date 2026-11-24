"""Check mixed yaw-frame force requests, the reachable segment and matched actor inputs."""
import numpy as np

from box_mpc import world_ee_target
from measure_force import nominal_targets, world_force
from run_world import command_schedule
from unifp_native import Rollouts, World


def check():
    scene = "artifacts/ee_study/general-upright_buffer/82/scene.mjb"
    live = World(scene, "artifacts/encoder_actor.onnx")
    live.run(command_schedule("push")[:50])
    root = live.base_pose
    rng = np.random.default_rng(317)
    n = 512
    roots = np.tile(root, (n,1))
    sph = rng.uniform([.1, -.95, -2.], [.65, 1.05, 2.], (n,3))
    u = np.zeros((n,1,9), np.float32)
    u[:,0,:3] = world_ee_target(roots, sph)
    u[:,0,6:] = rng.uniform(-40, 40, (n,3))
    pool = Rollouts(live, 4)
    snap = live.snapshot()
    b = pool.evaluate(snap,u)
    c = pool.evaluate(snap,u,matched_position=True)
    assert b["valid"].all() and c["valid"].all()
    b, c = b["policy_command"][:,0], c["policy_command"][:,0]
    assert not c[:,6:].any() and not b[:,9:].any()
    nominal = world_ee_target(roots, b[:,3:6])
    delta = world_force(roots, b)/200
    target = world_ee_target(roots, c[:,3:6])
    err = np.linalg.norm(target-nominal-delta, axis=1)
    assert err.max() < 5e-7
    # Check interior of segment as well: inner radius and pitch exclude nonconvex regions.
    for fraction in np.linspace(0., 1., 31):
        p = nominal + fraction*delta
        np.testing.assert_allclose(nominal_targets(roots, p), p, rtol=0, atol=3e-7)
    force = u[:,0,6:]
    scale = np.sum(b[:,6:9]*force,axis=1)/np.sum(force**2,axis=1)
    np.testing.assert_allclose(b[:,6:9], force*scale[:,None], atol=8e-6)
    assert scale.min() >= 0 and scale.max() <= 1.000001
    nclip = np.count_nonzero(scale < .9999)
    assert 0 < nclip < n
    # Commands just beyond the retained segment must violate at least one bound.
    desired = np.zeros((n,12))
    desired[:,6:9] = force
    ray = world_force(roots, desired)/200
    beyond = nominal + (scale+.001)[:,None]*ray
    short = scale < .998
    assert np.all(np.linalg.norm(nominal_targets(roots,beyond)-beyond,axis=1)[short] > 1e-8)
    reversed_batch = pool.evaluate(snap,u[::-1])
    np.testing.assert_array_equal(reversed_batch["policy_command"][::-1,0], b)
    print(f"{n} mixed XYZ requests, {nclip} shortened, max same-pose residual {err.max():.3g} m")


if __name__ == "__main__":
    check()
