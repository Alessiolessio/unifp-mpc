# Whole-body wall-force MPC

The robot can apply a sustained load while walking along a vertical face. The strongest result was on a simulated compliant surface. On the held-out 45 cm leftward walk, the force-conditioned controller traveled 44.1 cm, had 99.3% contact duty during the traverse and finished 1.1 cm from its base goal. Its moving force RMSE was 5.65 N at a 15 N setpoint, above the 4.5 N acceptance threshold. Ordinary position control also maintained contact and passed both stationary 15 N starts on that surface. These experiments establish a stationary capability and show where moving regulation still fails. [Walking video](../artifacts/wall_layer_tick/9/conditioned/episode.mp4) · [Stationary success](../artifacts/wall_layer_tick/8/off/episode.mp4) · [Low-load failure](../artifacts/wall_tick/0/conditioned/episode.mp4)

## Measurement, planning and evaluation

The fixed face is at world x = 0.89 m, 4 m wide and 1.6 m high, with friction 0.3. At each of four 5 ms physics steps the native controller sums finger and distal wrist contact forces on the wall. It rotates each solved MuJoCo wrench into world axes with the transpose of the contact frame, selects the sign for the wall's geom index and takes world +X as compression. Free steps contribute exactly zero. Integration over 20 ms divided by 20 ms gives the policy-tick load. We also retain the four individual loads for true substep peaks and duty above 1 N. The summed norm of all other robot–wall contact impulses is a separate safety measurement. On poses from the firm and compliant check episodes, every reconstructed hand-contact normal pointed along +X after accounting for geom order. [Native accumulation](../native/world.h) · [Trace binding](../native/world.cpp) · [Geometry and replay check](../tests/check_wall.py)

CEM branches from the executed state every five policy ticks. A optimizes world EE XYZ and body vx, vy and yaw rate. B and C add the same signed world-X scalar, with independent force noise, ±40 N request and 80 N/s request slew. B feeds the reachable force to UniFP in the instantaneous yaw frame. C encodes the same reachable displacement through the spherical position command with zero force input. UniFP scales force by 0.01 in its actor observation and its training reward shifts the nominal position by force / 200 N/m. Base force stays zero. All three use the same plant, initialization, XYZ and base search, three linear knots, 24 candidates, two elites, one iteration and eight workers. B and C have equal search dimension and share the same position workspace for the effective target. A has one fewer sampled scalar. [Force projection and training sources](ee_force_box.md) · [Sampler](../tools/cem.py) · [Wall planner](../tools/wall_mpc.py)

The default cost is the running mean of squared **relative measured load error**, plus squared terminal finger gap and squared terminal base XY error. Both distance terms use 12 cm as their unit. The gap is the positive face-to-finger-origin distance beyond 4 cm. It ranks candidate approaches before contact. The base target keeps its starting X and lies at most 12 cm down the wall toward its final Y goal. All base and arm commands are sampled. Native trajectory failures and predicted nongripper wall contact make a candidate ineligible.

Stationary runs last 5 s, walks 8 s, following an identical 1 s contact-free warmup across treatments. The development pose faces the wall. Held-out scene/CEM seeds 42/43 and 91/67 give initial X/yaw offsets of +0.9 cm/−1.1° and −2.5 cm/+7.1°. These are repeated measurements on three geometries. All force errors below include zeros. Whole-episode metrics include acquisition. Strict acceptance requires final-second tick RMSE ≤ max(1.5 N, 30% of target), at least 85% substep duty, base goal error ≤7.5 cm, tilt below 30° and zero unintended wall impulse. Walks also need 90% progress and must satisfy the same force RMSE and duty bounds during the first 10–90% of travel. The independent continuous half-second in-band check counts every tick. None of the trials passed that stronger measure.

## Matched results

Values in the first table are **whole-episode mean load / RMSE in N / duty in percent**. Each row uses the same start and independent replanning for A, B and C.

| Surface and task | A · position | B · conditioned | C · matched position |
|---|---:|---:|---:|
| Firm, stationary 3 N | 0.14 / 2.98 / 2 | <0.01 / 3.00 / <1 | 0.03 / 3.00 / <1 |
| Firm, stationary 5 N | 0.97 / 4.77 / 9 | 1.35 / 4.63 / 12 | 1.62 / 4.62 / 14 |
| Firm, stationary 8 N | 3.45 / 7.01 / 22 | 4.35 / 6.70 / 29 | 4.93 / 6.58 / 31 |
| Firm, held-out stationary 6 N | 3.90 / 5.24 / 24 | 2.29 / 5.50 / 18 | 3.12 / 5.36 / 21 |
| Firm, 45 cm right, 15 N | 13.03 / 8.44 / 64 | 13.45 / 7.06 / 61 | 12.79 / 6.39 / 62 |
| Compliant, 45 cm right, 15 N | 13.40 / 6.41 / 93 | 13.58 / 6.61 / 94 | 13.63 / 6.10 / 94 |
| Compliant, held-out 45 cm left, 15 N | 12.90 / 6.89 / 92 | 13.93 / 6.00 / 93 | 13.26 / 6.39 / 93 |

The firm-face default had **0/8 strict successes for each controller**, including the two held-out conditions. Low targets made a gap cheaper than many impacts. On the 5 N stationary start, first contact took 2.36 / 0.65 / 0.64 s for A/B/C, while peak 5 ms loads reached 27.8 / 43.1 / 34.1 N. Both B and C accelerated acquisition. On the held-out 5 N walk both ended with a completely unloaded final second. [Force traces](../artifacts/wall_force_traces.png) · [All per-episode measurements](wall_trials.csv)

A second **fixed** surface tested sensitivity to compliance after the firm-face failures. Only the face contact parameters changed, identically for predictions and live physics. The original solref is (0.02, 1) and solimp starts at (0.9, 0.95, 0.001). The compliant model uses (0.06, 1) and (0.005, 0.95, 0.05), with impedance increasing over 5 cm. Maximum overlap at checked policy-tick poses was 16.4 mm on the compliant held-out walk and 0.55 mm on the firm 5 N conditioned start. The material is a simulated sensitivity test without physical calibration. Two preliminary material settings in the CSV used solref/first three solimp values (0.08, 1)/(0.2, 0.6, 0.02) and (−240, −4)/(0.5, 0.5, 0.02). None met the strict criterion.

At **15 N stationary** the compliant face yielded the following **final-second mean / RMSE in N**. All six final-second duties were 100%. The success counts here describe two starts per mode.

| Start | A · position | B · conditioned | C · matched position |
|---|---:|---:|---:|
| Development | **15.30 / 4.19** | 16.83 / 5.25 | 15.59 / 5.01 |
| Held-out | **14.02 / 3.69** | **14.04 / 3.89** | 11.52 / 5.86 |

On the held-out compliant walk B acquired load at 0.54 s, stayed loaded for 7.45 of 8 s, had 4.48 N whole-episode MAE, 28.17 N peak, 3.54 cm maximum cross-track motion and 7.43° maximum tilt. The finger moved −29.0 cm along the wall. The final-second RMSE was 3.99 N. During travel, A/B/C RMSEs were 6.11 / 5.65 / 5.07 N. All three missed moving accuracy. B's RMS reward-target-to-finger separation was 7.73 cm, and its spherical command reached a bound on 72.8% of ticks. These data support coordinated motion with sustained loading on a compliant surface. They do not establish precise tracking throughout that motion.

## What the comparisons isolate

- **Force conditioning.** From identical pre-action states B and C reproduced the same reachable training target to within 0.087 µm on both checked episodes. Above 1 N of issued virtual force, the median maximum joint-action difference was 0.066 on the firm 5 N start and 0.030 on the held-out compliant walk. On the held-out stationary 15 N case, replaying B's entire schedule through C from the same warm state increased whole-episode RMSE from 6.57 to 13.29 N. Independent C replanning achieved 6.69 N, and A achieved 5.88 N. On the held-out walk the corresponding B / matched open-loop replay / independently optimized C errors were 6.00 / 5.99 / 6.39 N. B issued a signed mean virtual wall-directed force of +0.34 N during that walk while measured mean pressure force was 13.93 N. Replay recomputes the reachable target at each diverging base pose. Only the one-step fork isolates the actor at an identical state.
- **Alternative objective.** Scoring five 100 ms mean loads over the reference horizon and adding mean positive relative substep force deficit increased firm-face 5 N B duty from 12% to 24% and mean load from 1.35 to 3.42 N. It also increased RMSE from 4.63 to 5.27 N and peak load from 43.1 to 58.6 N. The tick objective remains the default because it penalizes this variation within the execution prefix.
- **Coordination and horizon.** Reducing the tangential base weight fourfold produced no strict walking successes. A 1 s horizon at fixed candidate count also gave no strict successes. On the development compliant 15 N walk it changed A/B/C moving RMSE from 4.69 / 6.26 / 5.79 N to 5.20 / 5.04 / 4.89 N and worsened final-second RMSE for all three. On the firm 5 N impulse-objective run, doubling the horizon reduced B duty from 24% to 17%.
- **Latency and parity.** The 0.5 s tick controller used 600 policy evaluations per plan. Firm-face plans over eight matched conditions had pooled p50/p95/max of 34.3 / 40.6 / 51.8 ms (1,470 updates). Compliant-face plans over six matched conditions took 35.7 / 43.3 / 55.8 ms (1,170 updates). The 1 s compliant walk used 1,200 evaluations and took 68.7 / 80.1 / 89.4 ms. Timings include the snapshot, sampling, native rollouts, cost, winner paths and warm start. Execution and rendering are outside the planning timer. Across the recorded comparisons there were 0/5,910 planning deadline misses. Selected command, action, pose, vector load, all four substep normal loads and unintended impulse agreed exactly for every executed prefix. Maximum tilt was 8.26° and unintended wall impulse was zero. The archived 8 cm box demo reproduced poses, velocities, actions, commands and gripper forces exactly. Box branching, viewer-path checks and cone contact restoration passed.

The comparison gives no consistent force-conditioning advantage. The next experiment should compare feasible force waveforms over a gait cycle at identical loaded states, then test a modest physical perturbation from a reproducibly regulated hold. External disturbance rejection remains unmeasured.

## Run

From the repository root with the existing model and scene assets prepared:

```bash
pixi run --locked view-wall --force 15 --padded --scenario-seed 42 --cem-seed 43 --episodes 1
pixi run --locked view-wall --force 15 --distance -.45 --padded --ee-force conditioned --scenario-seed 91 --cem-seed 67 --episodes 1
pixi run --locked view-wall --force 5 --ee-force matched
```

The viewer reports measured and desired newtons, shows the base goal and the winning full-horizon requested, projected and predicted finger paths, and repeats until closed unless an episode count is supplied. Results append to `artifacts/live_wall/episodes.jsonl`. Wall defaults are the firm face, 5 N, ordinary position MPC and the tick objective. For matched headless trials use the case indices in the runner, for example:

```bash
pixi run --locked env OPENBLAS_NUM_THREADS=1 PYTHONPATH=tools:build python tests/wall_trials.py 0 1 2 3 4 5 6 7 --out artifacts/wall_tick
pixi run --locked env OPENBLAS_NUM_THREADS=1 PYTHONPATH=tools:build python tests/wall_trials.py 6 7 8 9 --padded --out artifacts/wall_layer_tick
```

The CSV includes substep and tick error, acquisition, gaps, final-second holding accuracy, motion, stability, projections, matched replay and complete planning latencies per run. [Trial runner](../tests/wall_trials.py) · [Per-episode data](wall_trials.csv)
