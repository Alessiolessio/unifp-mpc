# Base-force conditioning during loaded locomotion

Ordinary velocity MPC settled all nine 90 cm walks, including two opposing-load magnitudes, an oblique walk with a lateral load, and an unannounced pulse. One-axis base-force MPC and its equal-dimensional matched-velocity control also settled nine. Ordinary velocity MPC had the lowest mean path error on the four held-out trials. The force input changed policy actions at identical states, but gave no repeatable task advantage at this planning budget. [Per-episode results](base_load_trials.csv)

## What the checkpoint was trained to do

The Spot actor receives yaw-frame virtual base force in newtons in raw policy command slots 9–11 (zero-based), scaled by 0.01 in actor frame entries 76–78. XY velocity commands are scaled by 2 and yaw rate by 0.25. The Spot overrides set virtual base force to ±30 N **per component** and independently scheduled external base force to ±25 N per component, with external Z multiplied by 0.1. At full curriculum strength the samples are uniform in each component. Virtual base pulses last 1–3 s between inactive intervals of 3.5–9 s. Physical base disturbances last 1–3 s between intervals of 6–12 s. The checkpoint's training notes put the end of the force ramp at iteration 4500 of 5300. [Command observation](../../unifp-spot/unifp_spot/envs/base/mdp/observations.py) · [Schedule defaults](../../unifp-spot/unifp_spot/envs/base/base_cfg.py) · [Spot overrides](../../unifp-spot/unifp_spot/envs/spot/spot_pos_force_cfg.py) · [Checkpoint provenance](../../unifp-spot/pretrained/spot_unifp/README.md)

The linear tracking reward uses the following horizontal target:

```text
u = v_cmd_xy + (F_virtual_yaw_xy + F_external_yaw_xy) / 200
active = |u_x| > 0.1 or |u_y| > 0.1 or |yaw_rate_cmd| > 0.2
v_target = u if active else [0, 0]
reward = exp(-||v_target - root_lin_vel_b_xy||² / 0.5²)   (weight 2)
```

The damping coefficient has units N·s/m. Yaw tracking has its own reward. The training command resampler also zeroes small nominal velocity triples. Several gait terms use the *nominal* walking mask. Matching the linear reward target therefore leaves room for different actor and gait responses. Z enters the observation and is sampled during training, but this reward only uses XY and provides no vertical force-tracking target. In a one-second unloaded check from a common warm state, ±20 N virtual Z changed the first action by up to 0.016–0.017 policy units. Final height changed by at most 2.1 mm at stand and 1.6 mm at a 0.3 m/s forward command. This small sensitivity provides no evidence for calibrated vertical control. [Reward and walking mask](../../unifp-spot/unifp_spot/envs/base/mdp/rewards.py) · [Command resampler](../../unifp-spot/unifp_spot/envs/unifp_env.py)

During training only the independent external stream writes a physical wrench. The current Isaac source yaw-rotates its stored vector and then passes the result to the wrench composer with `is_global=False` (link-frame interpretation). Its logged rotated vector is therefore insufficient to establish the actual world-frame load at arbitrary body orientation. Our MuJoCo experiment expressly applies a *world-fixed* horizontal force at the torso COM using `xfrc_applied`, with zero applied torque. This makes direction and magnitude reproducible through every 5 ms substep. A free-flight whole-system momentum check recovered (−18.975, +11.993) N for a requested (−19, +12) N over 20 ms. Snapshots include the wrench in `mjSTATE_INTEGRATION`. [Training force application](../../unifp-spot/unifp_spot/envs/unifp_env.py) · [Native setter](../native/world.cpp) · [Impulse and restore check](../tests/check_base_load.py)

## Controlled comparison

A, B and C optimize body vx, vy and yaw rate, with trained limits ±(0.6, 0.4, 0.6) in m/s, m/s and rad/s. The arm receives the same neutral yaw-relative spherical target and zero EE force throughout. A uses ordinary velocities. B additionally samples virtual force. C samples the **same** offset variables as B and issues ordinary velocity `v_cmd + F_virtual/200` with all force inputs zero. Both then have the same external contribution and deadband at a common physical state. Explicitly zeroing C's command inside the reward deadband would break that property in the presence of a load. All schedules use the native 12-slot command.

One axis is aligned with the known load in the initial yaw frame. It then rotates with Spot. For no-load and surprise trials it is chosen from the task direction alone. Two-axis runs add its planar perpendicular. Both treatments have the same 30 N magnitude cap and 80 N/s slew. Knots are restricted so **both** nominal and equivalent ordinary velocities stay within the trained bounds and 0.8 m/s² per-axis slew (0.8 rad/s² for yaw). This excludes targets beyond ordinary velocity authority. The maximum virtual XY offset available in the comparison is 0.15 m/s. Each extra force axis has a separate seeded noise stream.

Each update snapshots live physics, history and actuator state, evaluates 24 candidates for 25 policy ticks, fits the two elites once and executes five ticks. The planner uses the existing linear interpolation, elite fit and receding-horizon convention. Each compact rate ramps from the last issued value to the middle knot and holds at the third. The mean, variance and winning trajectory shift by 100 ms. Eight native workers share the ONNX session. The cost is half the sum of running-mean and terminal squared XY path error, plus mean squared heading chord error with a 0.20 m lever arm. The reference advances along the straight path at 0.30 m/s and holds at the goal. It specifies the task in world coordinates. All three base commands are selected by rollout cost. [Controller](../tools/base_load_mpc.py) · [Shared CEM](../tools/cem.py)

A common one-second unloaded warmup precedes each five-second controlled episode. Settling requires all final 0.5 s within 5 cm, mean-displacement speeds ≤0.10 m/s over each 100 ms prefix, and peak body tilt below 30°. We also record first qualifying hold, 2D velocity RMS error, lateral drift, heading, virtual force and goal error. Loads were known at the first planning snapshot except for the pulse. Development used spawn yaw 0° and CEM seed 17. Held-out goals were 35° from world X at spawn yaw 20° (CEM 43), and −35° at yaw −18° (CEM 67). The pulse and its unloaded twin share a spawn, goal and seed. One extra development run increased the reference pace to 0.45 m/s against 25 N to expose differences hidden by final settling. [Trial definitions](../tests/base_load_trials.py)

## Closed-loop results

Each cell gives **path RMS error in cm / final goal error in mm**, including acceleration and the final hold. All 27 primary episodes met the final settling criterion.

| World load and task | Ordinary velocity A | Conditioned B | Matched velocity C |
|---|---:|---:|---:|
| Free, forward | 3.63 / 2.2 | 3.76 / 4.3 | 3.60 / 4.6 |
| 15 N opposing | 5.77 / 3.2 | 6.06 / 4.5 | 5.87 / 2.3 |
| 25 N opposing | 7.86 / 5.0 | 8.26 / 0.8 | 7.87 / 0.4 |
| 18 N lateral | 3.78 / 4.3 | 3.85 / 0.4 | 3.88 / 2.2 |
| Held-out opposing (−19, −12) N | 6.60 / 2.3 | 7.29 / 1.3 | 6.86 / 2.0 |
| Held-out lateral (+12, +17) N | 3.20 / 3.7 | 3.58 / 3.1 | 3.43 / 1.0 |
| Held-out unloaded twin | 3.49 / 4.5 | 3.65 / 1.1 | 3.52 / 3.3 |
| Held-out pulse (−8, +22) N | 3.90 / 1.5 | 4.05 / 3.2 | 3.89 / 0.8 |
| Faster 0.45 m/s reference, 25 N opposing | 18.64 / 4.0 | 19.52 / 7.0 | 18.75 / 7.1 |

Held-out mean episode path RMS was **4.30 / 4.64 / 4.42 cm** for A/B/C. Mean episode velocity RMS error was **0.105 / 0.109 / 0.107 m/s**. At the ordinary pace, first settled holds completed at 3.6–3.7 s. At the faster pace, all three remained 30–32 cm short at the scheduled 2.0 s arrival, reached effective forward commands near 0.6 m/s and deviated in heading by 46–52°. They eventually settled at **4.5 / 4.8 / 4.9 s**. Across all 35 runs peak tilt was 9.66° and maximum cross-track error was 8.28 cm. The faster case exposes a timing and heading limitation even though its endpoint qualifies. [Faster loaded walk](../artifacts/base_load/fast25/conditioned/episode.mp4) · [25 N ordinary-velocity walk](../artifacts/base_load/opp25/off/episode.mp4)

**Unexpected load.** The 23.4 N pulse started at 1.04 s and ended at 1.64 s, both inside an executing prefix. Predictions held the latest load constant until the next snapshot. The loaded and unloaded twins had bit-identical poses before onset. Maximum XY separation from the respective unloaded twin was **4.68 / 4.33 / 4.50 cm**. After removal, A/B/C took **1.46 / 1.36 / 1.36 s** to confirm three consecutive full prefixes within 2 cm and 0.10 m/s of their twin (2.06 / 1.96 / 1.96 s from onset). Position and velocity recovery includes any divergence in subsequent MPC choices. This experiment assumes the new physical load becomes known at the next replan. [Pulse video](../artifacts/base_load/held_pulse/conditioned/episode.mp4) · [Paired recovery calculation](../tests/base_load_analysis.py)

**Conditioning and dimensionality.** Direct one-tick forks gave post-deadband reward targets equal to within 6.0e−8 m/s. For virtual magnitude above 1 N the per-run median maximum joint-action difference was 0.012–0.064 policy units across B runs. At 15 N opposing, B issued 11.38 N mean virtual magnitude, a 0.057 m/s mean pre-deadband shift. Replaying B's winning schedules with matched ordinary velocities from the same warmup gave **58.4 mm** final error versus **4.5 mm** for B. C replanned independently and finished at **2.3 mm**. On held-out opposing load the corresponding errors were **33.5 / 1.3 / 2.0 mm**. The actor uses the base-force channel, and a velocity-only optimizer can compensate for its absence.

Two horizontal coefficients had the same norm limit, independent draws and 24-candidate budget. The eight extra B/C runs all settled. B's path RMS with one → two coefficients changed from **3.85 → 3.95 cm** on the development lateral load, **3.58 → 3.43** on held-out lateral, **3.65 → 3.55** on the unloaded twin and **4.05 → 3.94** on the pulse. Ordinary velocity MPC reached **3.20 cm** on held-out lateral and **3.90 cm** on the pulse. The two-axis held-out lateral controller used 21.1 N mean virtual magnitude, compared with 8.5 N for one axis. These paired runs provide little basis for spending more optimizer dimensions on this task.

**Timing and fidelity.** Complete planning p50/p95/max were **30.7 / 36.9 / 45.2 ms**, with **0/1,750** misses of 100 ms on the eight available CPU cores. All **42,000** native candidates were valid. Timings include snapshot, sampling, native rollouts, cost, fit, winner selection and warm start. Playback, live stepping and offline probes are outside that clock. Every modeled tick reproduced executed qpos, qvel and action bit-for-bit (**8,720/8,720**). The 30 unmodeled ticks were the two pulse edges across five controllers. Maximum XY prediction error there was 1.27 mm. Box branching, cone contact restoration, and archived wall-normal loads also passed their existing checks.

## Reproduce

From the repo root with the prepared Spot model and exported policy:

```bash
pixi run --locked build
pixi run --locked env OPENBLAS_NUM_THREADS=1 PYTHONPATH=build python tools/base_load_mpc.py --mode conditioned --load -25 0
pixi run --locked env OPENBLAS_NUM_THREADS=1 PYTHONPATH=build:tools python tests/base_load_trials.py free opp15 opp25 side18 held_opp held_side held_free held_pulse fast25
pixi run --locked env OPENBLAS_NUM_THREADS=1 PYTHONPATH=build:tools python tests/base_load_trials.py side18 held_side held_free held_pulse --two
pixi run --locked env OPENBLAS_NUM_THREADS=1 PYTHONPATH=build:tools python tests/base_load_analysis.py artifacts/base_load docs/base_load_trials.csv
pixi run --locked env PYTHONPATH=build:tools python tests/check_base_load.py
```

The primary comparison uses one optimizer seed per pose, a flat floor, prescribed loads, a neutral arm and a generous endpoint hold. The tracking and pulse measurements are more discriminating than the success count. For a subsequent attached-box experiment, ordinary velocity MPC is the appropriate reference. First measure tether direction and tension and check that rollouts predict the slack-to-taut transition. Compare path lag at matched tension, achievable steady speed and recovery after the tether unloads. The current results give no reason to assume force conditioning will improve pulling.
