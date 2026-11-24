"""Walk to a planar destination under a world-frame torso load with UniFP MPC."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from forecasts import pack, winner
from box_mpc import latency, yaw_of
from cem import BASE_SLEW, DT, NODES, PREFIX, TIMES, fit, interpolate
from run_world import command_schedule
from unifp_native import Rollouts, World

SPEED = np.array([.6, .4, .6])
NEUTRAL = command_schedule("stand")[0, 3:6]
KD = 200.


def policy_commands(rates, basis, mode):
    """Yaw-frame velocities and virtual newtons -> native 12-slot policy commands."""
    cmd = np.zeros((*rates.shape[:-1], 12), dtype=np.float32)
    cmd[..., :3] = rates[..., :3]
    cmd[..., 3:6] = NEUTRAL
    if mode != "off":
        force = rates[..., 3:] @ basis
        if mode == "matched":
            cmd[..., :2] = rates[..., :2] + force / KD
        else:
            cmd[..., 9:11] = force
    return cmd


def effective_velocity(cmd, external_body_xy):
    external = np.asarray(external_body_xy, dtype=cmd.dtype)
    vel = cmd[..., :2] + (cmd[..., 9:11] + external) / KD
    walking = np.any(np.abs(vel) > .1, axis=-1) | (np.abs(cmd[..., 2]) > .2)
    return vel * walking[..., None]


def sample_rates(mean, sigma, winner, previous, scale, basis, rng, force_rngs, count,
                 nodes=NODES, broad_rng=None):
    # Same three linear knots as the mobile planner, one destination rate per axis.
    noise = rng.standard_normal((count, 3))
    if len(basis):
        noise = np.column_stack([noise] + [r.standard_normal(count) for r in force_rngs])
    end = mean[1] + sigma[1] * noise
    if broad_rng is not None:
        n = max(1, count // 4)
        end[-n:, :3] = broad_rng.uniform(-1., 1., (n, 3))
    end[:2] = [winner[1], mean[1]]
    end = np.clip(end, -1., 1.) * scale
    start = previous * scale
    dt = nodes[1]
    end[:, :3] = np.clip(end[:, :3], start[:3]-BASE_SLEW*dt, start[:3]+BASE_SLEW*dt)
    if len(basis):
        # A common 30 N disk for one or two dimensions. Bound the equivalent ordinary
        # command and its slew too, so conditioning buys neither speed nor acceleration range.
        f = end[:, 3:]
        f *= np.minimum(1., 30. / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12))
        df = f - start[3:]
        f[:] = start[3:] + df * np.minimum(
            1., 80.*dt / np.maximum(np.linalg.norm(df, axis=1, keepdims=True), 1e-12))
        old = start[:2] + start[3:] @ basis / KD
        change = end[:, :2] + f @ basis / KD - old
        room = np.minimum(BASE_SLEW*dt, np.where(change >= 0, SPEED[:2]-old, SPEED[:2]+old))
        ratio = np.divide(room, abs(change), out=np.ones_like(room), where=abs(change) > 1e-12)
        fraction = np.clip(np.min(ratio, axis=1), 0., 1.)
        end[:, :2] = start[:2] + fraction[:, None] * (end[:, :2]-start[:2])
        f[:] = start[3:] + fraction[:, None] * (f-start[3:])
    knots = np.repeat((end / scale)[:, None, :], len(nodes), axis=1)
    knots[:, 0] = previous
    return knots


class RateCEM:
    """Compact body-rate knots, with optional paired virtual-force coefficients."""

    def __init__(self, mode, basis, seed, horizon=.5, broad=False):
        self.nodes = NODES * (horizon/.5)
        self.times = np.arange(round(horizon/DT))*DT
        self.mode, self.basis = mode, basis
        self.scale = np.r_[SPEED, np.full(len(basis), 30.)]
        self.previous = np.zeros(len(self.scale))
        self.mean = np.zeros((len(NODES), len(self.scale)))
        self.winner = self.mean.copy()
        self.sigma = np.full_like(self.mean, .5)
        self.rng = np.random.default_rng(seed)
        self.broad_rng = None
        if broad:
            stream = max(2, len(basis)+1)
            self.broad_rng = np.random.default_rng(np.random.SeedSequence([seed, stream]))
        self.force_rngs = [np.random.default_rng(np.random.SeedSequence([seed, i+1]))
                           for i in range(len(basis))]

    def sample(self):
        self.knots = sample_rates(self.mean, self.sigma, self.winner, self.previous, self.scale,
                                  self.basis, self.rng, self.force_rngs, 24, self.nodes, self.broad_rng)
        self.rates = interpolate(self.knots, self.times, self.nodes) * self.scale
        return policy_commands(self.rates, self.basis, self.mode)

    def update(self, cost):
        mean, sigma, best = fit(self.knots, cost)
        self.previous = self.rates[best, PREFIX-1] / self.scale
        self.mean, self.sigma, self.winner = (interpolate(x, self.nodes + PREFIX*DT, self.nodes)
                                               for x in (mean, sigma, self.knots[best]))
        return best


def reference(origin, delta, t, pace):
    return origin + np.minimum(pace * t / np.linalg.norm(delta), 1.)[..., None] * delta


def episode(live, pool, mode, load, delta, seed, seconds, pulse, dimensions, folder, scene, pace=.30):
    warm = live.run(command_schedule("stand")[:50])
    origin = live.base_pose[:2]
    heading = yaw_of(live.base_pose[3:7])
    goal = origin + delta
    # Co-rotating UniFP axes. For unannounced pulses use the task line even before onset,
    # exactly as in the no-load control. The sampling basis never reveals a future shove.
    axis = load if np.any(load) and not pulse else delta
    c, s = np.cos(heading), np.sin(heading)
    axis = np.array([c*axis[0]+s*axis[1], -s*axis[0]+c*axis[1]]) / np.linalg.norm(axis)
    basis = np.array([axis, [-axis[1], axis[0]]])[:0 if mode == "off" else dimensions]
    planner = RateCEM(mode, basis, seed)
    # A sustained known load is set before the first snapshot. Pulse edges occur two
    # ticks into a prefix. Until the next replan every forecast retains its old load.
    def applied(tick):
        return load if not pulse or 52 <= tick < 82 else np.zeros(2)

    live.set_torso_load(*applied(0))
    winners = []
    history, plans, times, validity = [], [], [], []
    physical, predicted, modeled = [], [], []
    for start in range(0, round(seconds / DT), PREFIX):
        begin = perf_counter()
        snapshot = live.snapshot()
        commands = planner.sample()
        pre_root = live.base_pose
        forecasts = pool.evaluate(snapshot, commands)
        ref = reference(origin, delta, start*DT + TIMES + DT, pace)
        error = np.sum((forecasts["qpos"][..., :2] - ref)**2, axis=-1)
        yaw = yaw_of(forecasts["qpos"][..., 3:7]) - heading
        # Heading deviation in metres at a 20 cm lever arm.
        cost = .5*(error.mean(1) + error[:, -1]) + .2**2 * (2.-2.*np.cos(yaw)).mean(1)
        cost[~forecasts["valid"]] = np.inf
        validity.append(int(np.isfinite(cost).sum()))
        if not validity[-1]:
            break
        best = planner.update(cost)
        prefix = commands[best, :PREFIX].copy()
        prediction = {k: forecasts[k][best, :PREFIX].copy() for k in ("qpos", "qvel", "action")}
        selected = winner(forecasts, best, commands, pre_root)
        selected["reference_xy"] = ref
        selected["load_world_N"] = np.broadcast_to(applied(start), (len(TIMES), 2))
        winners.append(selected)
        times.append(perf_counter() - begin)
        plans.append(prefix)
        predicted.append(prediction["qpos"][:, :2])
        consistent = True
        for j, cmd in enumerate(prefix):
            current = applied(start+j)
            live.set_torso_load(*current)
            physical.append(current)
            consistent = consistent and np.array_equal(current, applied(start))
            modeled.append(consistent)
            trace = live.run(cmd[None])
            if consistent:
                for key, value in prediction.items():
                    np.testing.assert_array_equal(trace[key][0], value[j])
            history.append(trace)
    if not history:
        raise RuntimeError("No valid plan on the initial snapshot")
    traj = {k: np.concatenate([x[k] for x in history]) for k in ("qpos", "qvel", "action", "ee_world")}
    cmds = np.concatenate(plans)
    physical = np.asarray(physical)
    n = len(cmds)
    pose = traj["qpos"]
    t = (np.arange(n)+1)*DT
    xy = pose[:, :2]
    line = delta / np.linalg.norm(delta)
    ref = reference(origin, delta, t, pace)
    xy0 = np.vstack((origin, xy))
    # Prefix-average displacement velocity rejects intra-gait oscillations in settling metrics.
    vel = np.diff(xy0[::PREFIX], axis=0) / (DT*PREFIX)
    ref_vel = np.diff(np.vstack((origin, ref))[::PREFIX], axis=0) / (DT*PREFIX)
    err = np.linalg.norm(xy-goal, axis=1)
    stationary = (err.reshape(-1,PREFIX).max(1) <= .05) & (np.linalg.norm(vel, axis=1) <= .10)
    holds = np.flatnonzero(np.convolve(stationary, np.ones(5), "valid") == 5) if len(vel)>=5 else []
    x, y = pose[:, 4:6].T
    tilt = np.rad2deg(np.max(np.arccos(np.clip(1-2*(x*x+y*y), -1, 1))))
    yaw_error = yaw_of(pose[:, 3:7])-heading
    pred_error = np.linalg.norm(xy-np.concatenate(predicted), axis=1)
    progress = (xy-origin) @ line
    force = np.linalg.norm(cmds[:, 9:11], axis=1)
    result = dict(task="base_load", controller=mode, dimensions=len(basis), seed=seed, pace_m_s=pace,
                  scene=str(scene), goal_xy_m=goal.tolist(), delta_m=delta.tolist(),
                  heading_rad=float(heading), load_world_N=load.tolist(), pulse=pulse,
                  simulated_s=float(t[-1]), success=bool(len(stationary)>=5 and stationary[-5:].all()
                      and tilt < 30 and n == round(seconds/DT)), final_error_m=float(err[-1]),
                  hold_s=float((holds[0]+5)*PREFIX*DT) if len(holds) else None,
                  progress_m=float(progress[-1]), path_rmse_m=float(np.sqrt(((xy-ref)**2).sum(1).mean())),
                  velocity_rmse_m_s=float(np.sqrt(((vel-ref_vel)**2).sum(1).mean())),
                  cross_track_max_m=float(np.abs((xy-origin) @ [-line[1], line[0]]).max()),
                  heading_max_deg=float(np.rad2deg(abs(np.arctan2(np.sin(yaw_error),np.cos(yaw_error))).max())),
                  tilt_max_deg=float(tilt), virtual_mean_N=float(force.mean()), virtual_max_N=float(force.max()),
                  forecast_max_m=float(pred_error.max()), expected_exact_ticks=int(np.sum(modeled)),
                  valid_candidates=sum(validity), total_candidates=24*len(validity), **latency(times))
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder/"episode.npz", **traj, **pack(winners, 0),
                        initial_qpos=warm["qpos"][-1], time_s=t, policy_command=cmds,
                        load_world_N=physical, reference_xy=ref, reference_vel=ref_vel,
                        prefix_vel=vel, prediction_xy=np.concatenate(predicted), exact=modeled,
                        latency_s=times, origin=origin)
    (folder/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["off", "conditioned", "matched"], default="off")
    p.add_argument("--load", nargs=2, type=float, default=[-15., 0.], metavar=("FX", "FY"))
    p.add_argument("--delta", nargs=2, type=float, default=[.9, 0.], metavar=("DX", "DY"))
    p.add_argument("--dimensions", choices=[1, 2], type=int, default=1)
    p.add_argument("--pulse", action="store_true", help="Unannounced physical load from 1.04 to 1.64 s")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--pace", type=float, default=.30, help="Reference speed in m/s")
    p.add_argument("--seconds", type=float, default=5.)
    p.add_argument("--scene", type=Path, default=Path("artifacts/spot.mjb"))
    p.add_argument("--out", type=Path, default=Path("artifacts/base_load/demo"))
    args = p.parse_args()
    if (not np.linalg.norm(args.delta) or args.pace <= 0 or args.seconds < .5
            or not np.isclose(args.seconds*10, round(args.seconds*10))):
        p.error("Supply a nonzero destination, positive pace and whole 0.1 s periods (at least 0.5 s)")
    live = World(str(args.scene), "artifacts/encoder_actor.onnx")
    result = episode(live, Rollouts(live, 8), args.mode, np.array(args.load), np.array(args.delta),
                     args.seed, args.seconds, args.pulse, args.dimensions, args.out, args.scene, args.pace)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
