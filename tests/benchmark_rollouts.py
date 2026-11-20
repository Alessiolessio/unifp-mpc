"""Warm snapshot-to-NumPy latency for one 0.50 s candidate batch in box contact."""

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
from run_world import command_schedule
from unifp_native import Rollouts, World


def schedules(root):
    w, x, y, z = root[3:7]
    yaw = np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    anchor = root[:3] + [.83 * np.cos(yaw), .83 * np.sin(yaw), -.08]
    rng = np.random.default_rng(0)
    knots = np.broadcast_to(anchor, (32, 3, 3)).copy()
    knots[1:, 1:] += rng.uniform(-.035, .035, size=(31, 2, 3))
    assert np.linalg.norm(np.diff(knots, axis=1), axis=2).max() <= .5 * .25
    times = np.arange(25) * .02
    return np.array([[np.interp(times, [0, .25, .5], k) for k in xyz.T]
                     for xyz in knots], dtype=np.float32).transpose(0, 2, 1).copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    live = World("artifacts/spot_box.mjb", "artifacts/encoder_actor.onnx")
    prefix = live.run(command_schedule("push")[:93])
    assert prefix["contact_duty"][-1] > 0
    targets = schedules(prefix["qpos"][-1])
    snapshot = live.snapshot()
    reference = Rollouts(live, 1).evaluate(snapshot, targets)
    assert reference["valid"].all() and (reference["contact_duty"].max(axis=1) > 0).all()
    advance = reference["qpos"][:, -1, -7] - prefix["qpos"][-1, -7]
    print(f"32 contact trajectories, box forward displacement {advance.min():.4f}–{advance.max():.4f} m")
    print(f"{len(os.sched_getaffinity(0))} logical CPUs available")
    print("workers  batch    p50    p95    max   misses    cores  ORT/phys ms per candidate  dispatch ms")
    rows = []
    for workers in (1, 4, 8):
        pool = Rollouts(live, workers)
        idle_cpu = time.process_time()
        idle_start = time.perf_counter()
        time.sleep(.25)
        idle_cores = (time.process_time() - idle_cpu) / (time.perf_counter() - idle_start)
        print(f"{workers} workers idle for 250 ms: {idle_cores:.4f} cores")
        check = pool.evaluate(snapshot, targets)
        for k in prefix:
            np.testing.assert_array_equal(check[k], reference[k])
        for count in (1, 8, 16, 32):
            inputs = targets[:count]
            pool.evaluate(snapshot, inputs)
            samples = []
            for repeat in range(args.repeats):
                cpu = time.process_time()
                start = time.perf_counter()
                state = live.snapshot()
                after_snapshot = time.perf_counter()
                result = pool.evaluate(state, inputs)
                end = time.perf_counter()
                cpu = time.process_time() - cpu
                assert result["valid"].all()
                samples.append(dict(workers=workers, candidates=count, repeat=repeat,
                                    latency_ms=1000*(end-start), snapshot_ms=1000*(after_snapshot-start),
                                    cpu_ms=1000*cpu, idle_cores=idle_cores,
                                    inference_work_ms=1000*result["inference_work_s"],
                                    physics_work_ms=1000*result["physics_work_s"],
                                    dispatch_ms=1000*result["dispatch_s"]))
            rows.extend(samples)
            wall = np.array([r["latency_ms"] for r in samples])
            cpu_cores = sum(r["cpu_ms"] for r in samples) / wall.sum()
            infer = np.mean([r["inference_work_ms"] for r in samples]) / count
            physics = np.mean([r["physics_work_ms"] for r in samples]) / count
            dispatch = np.median([r["dispatch_ms"] for r in samples])
            print(f"{workers:7d} {count:6d} {np.median(wall):6.1f} {np.percentile(wall, 95):6.1f}"
                  f" {wall.max():6.1f} {sum(wall > 100):3d}/{len(wall):<3d} {cpu_cores:7.2f}"
                  f" {infer:11.2f}/{physics:<9.2f} {dispatch:.3f}", flush=True)
        del pool
    if args.csv:
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
