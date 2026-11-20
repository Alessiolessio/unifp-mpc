# Randomized box pushing

The 24-candidate controller reduced average final XY error and planning time on both development and held-out starts. Settling success was equal on each split. The original 32-candidate mobile controller remains available for the nominal demonstrations. [Per-episode measurements](box_trials.csv)

| Split and controller | Settled | Mean final error, all trials | Planning median / p95 / max | Planning misses | Full cycle p95 |
|---|---:|---:|---:|---:|---:|
| Development, original | 2/4 | 28.5 mm | 53.6 / 63.0 / 90.7 ms | 0/236 | 65.1 ms |
| Development, selected | 2/4 | 17.1 mm | 35.1 / 42.0 / 47.8 ms | 0/281 | 44.0 ms |
| Held out, original | 3/4 | 25.8 mm | 43.3 / 52.2 / 56.4 ms | 0/177 | 54.0 ms |
| Held out, selected | 3/4 | 17.6 mm | 32.3 / 39.7 / 49.2 ms | 0/180 | 41.6 ms |

In the CSV, goal offsets and base displacements are world metres. Contact ticks count 20 ms intervals with gripper force above 1 N in at least one substep. The longest gap between contacts and the final gap after contact loss are reported separately. Cross-track and overshoot are peak deviations over the executed episode. Workspace fractions count issued positions within 1 cm of, or beyond, the initial XY bounds and projected radii at the 0.55 m cap.

Each update branches all candidates from one executed snapshot. Planning times include sampling, all native forecasts, scoring and elite updates. The synchronous cycle also executes five live policy ticks. Percentiles pool individual updates on the eight-vCPU Xeon. The four rows had zero full-cycle misses of 100 ms. Each episode starts with one second of contact-free warmup. Success requires 25 consecutive 50 Hz samples within 15 mm of the final goal and at most 2 cm/s box XY speed for the last five. Timeouts at nine control seconds contribute to every error average.

## Scenarios and contact

Robot XY is uniform within ±12 cm and yaw within ±15°. The box starts 1.035–1.105 m ahead and within ±8.5 cm sideways in the initial yaw frame, with up to ±20° relative yaw. Goal bearings span ±30° in that frame. The normal displacement range is 12–38 cm and the longer range is 48–70 cm. The runner adds the world-rotated displacement to the **executed box center after warmup**. The generator checks for robot–box contact at spawn and the runner checks the warmup. Scenario and optimizer RNGs are independent.

The box geom has priority 6 and the ground has priority 5. The original box friction therefore controlled both box–ground and gripper–box contacts. Randomized scenes use one explicit ground–box pair for sliding friction in [0.20, 0.46]. The box geom controls hand–box sliding friction independently in [0.22, 0.50]. The pair carries the box contact dimension, solver reference, impedance, torsional and rolling coefficients of the original contact. Replaying executed poses and inspecting the two tangential coefficients in actual contacts verified both values. Predictors share the live model in every trial. These experiments measure matched-physics sensitivity. [Scenario generator](../tools/box_scenario.py) · [Scene preparation](../tools/prepare_scene.py) · [MuJoCo 3.6 contact combination rules](https://mujoco.readthedocs.io/en/3.6.0/modeling.html#contact-parameters)

## Cost and search

SUMO's simplified Spot task averages goal and gripper-to-center distances over time and penalizes squared mean object speed, with weights 60, 4 and 20. Its Spot overrides use a 2 s horizon, 24 samples, three knots, three CEM elites and a 3.5 noise ramp. Its box task inherits Judo's wider reset distribution, which samples an annulus, adds Gaussian object XY noise and independently samples robot XY with identity orientation. UniFP uses a directional rear-face target, bounded body-forward speed and EE slew, a box-centered workspace, and gripper-only manipulation. Its local box goal advances up to 8 cm along the push line. [SUMO reward](../../sumo/sumo/tasks/spot/spot_push.py) · [Controller overrides](../../sumo/sumo/controller/overrides.py) · [Optimizer overrides](../../sumo/sumo/controller/optimizer_overrides.py) · [Pinned Judo reset](../../sumo/.judo-src/judo/tasks/spot/spot_box_push.py) · [Judo CEM](../../sumo/.judo-src/judo/optimizers/cem.py)

In the reference development failures, case 0 drifted 64 mm off line and lost gripper contact for the final 5.84 s. Case 3 drifted 114 mm off line and spent 21% of issued requests within 1 cm of a world XY bound. Moving the bounds with the executed box produced 0/2 successes on these cases. Adding a running cross-track penalty reduced peak drift on both but also produced 0/2. Mixing mean and terminal box error equally recovered case 0 at 32 samples but missed case 3 by 83 mm.

The selected objective keeps that equal mixture. It bounds the tradeoff against the original combined approach and base-spacing cost `G` by replacing it with `B × G / (B + G)`, where `B` is the larger of squared current local-goal distance and squared settling radius. This preserves a ranking before contact and limits the value of improving finger placement once the box is close. Falls, nonfinite forecasts and predicted nongripper impulses remain ineligible. At 24 candidates, the original terminal cost settled 1/4 development cases with 37.8 mm mean error. The bounded cost settled 2/4 at 17.1 mm. A running speed penalty with the unbounded running cost settled 0/4. Adding SUMO's noise ramp and third elite to that speed-cost run also settled 0/4. A separate terminal-speed penalty when the final goal became active gave 39.6 and 20.2 mm on development cases 0 and 2, versus 11.7 and 17.4 mm with the selected cost. [Planner](../tools/box_mpc.py) · [Knot sampling and interpolation](../tools/cem.py)

The horizon comparison used the same running cost, development cases 0 and 3, seed 0, three equally spaced knots and a 0.1 s prefix. The warm start shifts by that actual prefix. Both speed and EE slew limits use the physical knot spacing. Work counts UniFP evaluations per update, with four physics steps per evaluation.

| Candidates × iterations | Horizon | Policy evaluations | Settled | Mean final error | Planning p95 / max | Planning misses |
|---|---:|---:|---:|---:|---:|---:|
| 16 × 1 | 0.5 s | 400 | 0/2 | 55.2 mm | 31.3 / 37.4 ms | 0/180 |
| 24 × 1 | 0.5 s | 600 | 0/2 | 70.7 mm | 44.3 / 52.6 ms | 0/180 |
| 32 × 1 | 0.5 s | 800 | 1/2 | 45.9 mm | 71.1 / 108.6 ms | 1/157 |
| 16 × 2 | 0.5 s | 800 | 0/2 | 29.3 mm | 66.3 / 71.3 ms | 0/180 |
| 16 × 1 | 1.0 s | 800 | 0/2 | 82.5 mm | 62.5 / 69.7 ms | 0/180 |
| 16 × 3 | 0.5 s | 1,200 | 0/2 | 47.5 mm | 89.7 / 100.3 ms | 1/180 |
| 16 × 1 | 1.5 s | 1,200 | 1/2 | 43.8 mm | 85.4 / 106.2 ms | 2/176 |

The selected configuration uses fixed bounds, two elites, three knots, 24 × 1 and 0.5 s. The 600 policy evaluations save 25% of rollout work per update. An additional sampled body-lateral speed with the bounded cost also settled 2/4 development tasks, at 19.4 mm mean error. Exploratory lateral stance feedback recovered both original failures in one configuration but lost both original successes. The CSV includes unsuccessful variants. Labels beginning with `stance`, `choose`, and `follow` denote imposed alignment, sampled alignment on/off, and initial-stance tracking. `guided24` tested rear-face-centered proposals in a quarter of the population. These probes were excluded from the selected controller.

## Held-out outcomes and limits

The configuration was fixed before evaluating seeds 11–14 with paired CEM seeds 17, 29, 17, 29. Matching warmup states, actions and goals were checked exactly. Times below exclude warmup.

| Scenario and goal displacement, world XY | Original error / time | Selected error / time |
|---|---:|---:|
| 11, (+14.2, −5.9) cm | 9.5 mm / 2.1 s | 8.8 mm / 2.1 s |
| 12, (+15.0, +0.2) cm | 3.2 mm / 4.5 s | 1.4 mm / 4.8 s |
| 13, (+11.6, +3.4) cm | 1.6 mm / 2.1 s | 9.5 mm / 2.1 s |
| 14, (+59.2, +23.7) cm | 88.7 mm / timeout | 50.7 mm / timeout |

On case 14 the selected controller advanced the base 68 cm along world X and recorded 172 gripper-contact ticks, but peak lateral deviation was 124 mm. It spent 24% of issued requests within 1 cm of a world XY bound. Both held-out controllers stayed upright, with peak absolute roll/pitch below 8.5°, zero nongripper impulse and exact selected-prefix reproduction. The trial count supports a latency gain and a smaller miss on this long push. It is too small to establish a success-rate gain. [Settled episode](../artifacts/robust_box/held_selected/12/episode.mp4) · [Long-push timeout](../artifacts/robust_box/held_selected/14/episode.mp4)

On development case 1 with the same pose, goal and optimizer seed, raising ground friction alone from 0.343 to 0.46 changed selected error from 3.6 to 13.9 mm. Raising hand friction alone from 0.228 to 0.50 gave 8.3 mm. All three settled. A separate 45 cm pure sideways goal missed by 406.5 mm with forward speed and by 185.5 mm with sampled lateral speed. This probes the control-space limit outside the sampled forward sector. Neither run establishes that the pose is mechanically unreachable.

The original 8, 35 and 50 cm demos replayed their archived states, actions, contacts and plan requests exactly. The selected configuration finished at 2.8, 19.1 and 10.0 mm respectively, missing the 35 cm tolerance. Use the reference for reproducing those demonstrations. The next focused experiment is a recovery maneuver after sustained lateral slip, with paired resets at the first loss of useful contact. A live/predictor friction mismatch remains a separate experiment.

## Reproduction

Run from the repository root after preparing the native runtime and local assets. The study runner creates missing scenes, evaluates the normal controller entry point, checks prefix interpolation and actual settling, and writes per-run traces plus a metrics CSV. Reuse the same scenario directories for paired comparisons. Within `tools/box_scenario.py`, `--ground-mu` and `--hand-mu` override one coefficient while preserving the pose and random draws. `--goal-body DX DY` specifies box displacement in the initial robot yaw frame.

```bash
pixi run --locked build
export PYTHONPATH=build:tools OPENBLAS_NUM_THREADS=1
pixi run --locked python tests/study_box.py reference --cases 0 1 2 3
pixi run --locked python tests/study_box.py bounded_forward --cases 0 1 2 3 --objective bounded --candidates 24
pixi run --locked python tests/study_box.py held_reference --cases 11 12 13 14 --cem-seeds 17 29 17 29
pixi run --locked python tests/study_box.py held_selected --cases 11 12 13 14 --cem-seeds 17 29 17 29 --objective bounded --candidates 24
```
