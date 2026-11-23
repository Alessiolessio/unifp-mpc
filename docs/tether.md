# Tethered box transportation

Ordinary velocity MPC transported the box to four of five 80 cm goals with a 1 s horizon. Two-axis virtual base force and its matched ordinary-velocity control each settled the same four. Both made considerably more progress than ordinary MPC on the fifth start. Ordinary velocity remains the default. [Trial results](tether_trials.csv) · [Diagonal success](../artifacts/tow_1s/diagonal/off/episode.mp4) · [Side-goal timeout](../artifacts/tow_1s/side/conditioned/episode.mp4)

## Physical model and checks

The calibrated Spot tows the existing free 40 × 40 × 50 cm cuboid. A two-site spatial tendon connects the upper rear edge of the torso at local (−0.42, 0, 0.08) m to the box's front face at (0.20, 0, 0.20) m. A unilateral MuJoCo length limit of 0.70 m supplies the entire load. The cable has zero passive stiffness, damping, friction and actuator force. Its lower limit of zero is geometrically unreachable on these pulls. Changing mass scales all three principal box inertias at constant dimensions. The explicit box–floor contact pair sets both sliding friction directions. All cases start clear of Spot, with the requested spare length, and complete one second of neutral warmup with zero towing or robot–box impulse. [Scene construction](../tools/prepare_scene.py) · [MuJoCo spatial tendons](https://mujoco.readthedocs.io/en/3.6.0/XMLreference.html#tendon-spatial)

At each 5 ms physics step the native controller reads the cable's unilateral solver multiplier in newtons, or zero while inactive. At the box, tension acts along the world vector from eye to hitch. In free flight the measured box momentum change after subtracting gravity was (0.120785, 0.000004, 0.022222) N·s. The integrated tension along the initial cable direction gave (0.120795, 0, 0.022168) N·s. With the cable disconnected, replaying the winning straight and diagonal commands displaced the box by less than 10⁻¹⁵ m. All executed primary trials had zero direct robot–box impulses. Sampled cable stretch at 50 Hz stayed below 3 mm. [Native measurement](../native/world.h) · [Impulse, disconnect and branch checks](../tests/check_tether.py)

A first diagonal prototype let the taut cable pass through a hind leg on 113 policy ticks. The final model raises both attachment sites and ray-checks the straight cable centreline against the collision geometry on every substep with positive tension. CEM rejects a candidate if any such segment enters a collision geom, if the robot directly contacts the box, or if the native rollout fails. Executed prefixes contained zero flagged cable crossings. This is a massless cable model for unobstructed towing. The slack catenary is visual and the solver does not simulate snagging, winding, distributed cable mass or finite-thickness contacts. The reported slack/taut transitions count crossings of 0.1 N at 200 Hz and include small gait-driven unloadings.

## Planner and comparisons

Each update branches from the executed MuJoCo and UniFP state, including solver warm-start, policy history and actuator queues. The shared rate sampler draws 24 candidates with three linear knots, fits two elites in one CEM iteration and shifts its distribution and winner by the 0.1 s execution prefix. Eight native workers run the 50 Hz policy and 200 Hz physics. CEM chooses all three yaw-frame rates. The EE receives the same neutral spherical command and zero force. [Planner](../tools/tether_mpc.py) · [Shared rate sampler](../tools/base_load_mpc.py)

For planned box XY positions, the principal cost is half the sum of running-mean and terminal squared error to a local reference advancing from the current box toward the world goal at 0.23 m/s, capped at the goal. This pace is a box-motion reference. All robot movements come from evaluated policy commands. In an initial straight-pull test, scoring the distant endpoint directly led to a peak box speed of 0.79 m/s and 53 cm peak cross-track error. A terminal cable-direction term gives candidates a ranking while the box is motionless. Its target is a taut eye-to-hitch vector pointing along the current box-goal direction, with horizontal length corrected for predicted attachment height difference. Its weight on squared XY vector error is 0.15 × min(1, remaining distance / 0.15 m)². There is no force-command reward, prescribed robot heading, or pulling sequence. Success requires 25 consecutive policy ticks within 5 cm at box XY speed ≤0.08 m/s, confirmed at a prefix boundary. The episode limit is 9 s after warmup.

A uses ordinary velocity. B samples two additional horizontal force coefficients in the instantaneous torso yaw frame. C samples the same two coefficients and issues ordinary velocity shifted by F/200, with virtual force zero. B/C share the same seeded velocity/force proposals on a common initial distribution, trained velocity bounds, 30 N force norm, 80 N/s force slew, and 0.8 per-axis velocity/yaw slew. The nominal and shifted commands must both be feasible. At a common physical state the external contribution and training deadband are identical for B and C. The tether's physical response evolves independently in each predicted trajectory. [UniFP reward and comparison details](base_load.md)

## Matched episodes

Each cell gives **final box error in cm / simulated seconds after warmup** at the 1 s horizon. All three modes use identical physical starts and optimizer seed per row. The first three cases were used for development, the final two for comparison on new starts. There is one primary optimizer seed per case.

| Box, goal bearing, initial torso yaw | Floor μ / spawn slack | A ordinary | B conditioned | C matched |
|---|---|---:|---:|---:|
| 1 kg, straight, 0° | 0.30 / 4 cm | 3.44 / 4.8 | 0.12 / 4.8 | 0.57 / 4.8 |
| 1 kg, +30°, 0° | 0.30 / 8 cm | 2.67 / 6.3 | 3.36 / 6.4 | 2.89 / 6.3 |
| 3 kg, straight, +15° | 0.35 / 12 cm | 3.01 / 5.7 | 2.79 / 5.1 | 2.40 / 5.1 |
| 2 kg, −38°, +18° | 0.27 / 10 cm | 1.47 / 5.1 | 3.11 / 5.2 | 2.89 / 5.2 |
| 3 kg, +65°, −15° | 0.40 / 14 cm | 80.00 / 2.0, no valid plan | 5.24 / 9.0, timeout | 6.05 / 9.0, timeout |

On the 30° pull A moved the box 81.7 cm along the goal axis, with 8.1 cm maximum lateral deviation and 0.27 m/s peak speed. Spot displaced 58.7 cm and turned through 37°. The loaded cable swung through 75.8°, carrying 1.92 N averaged across the whole trial, 16.96 N peak and 75.4% loaded duty. On the 3 kg straight case A averaged 7.63 N, peaked at 60.18 N and had 82.2% duty. Across the fifteen 1 s trials, maximum body tilt was 9.35°.

B used mean virtual magnitudes of 8.6–23.5 N in the five 1 s episodes. On the diagonal trial its mean reward-target shift was 0.103 m/s. At the peak-load policy tick, matching the velocity target from the same history changed the maximum joint action by 0.0544 policy units. Replaying the matched commands without replanning ended 11.17 cm from goal. B finished 3.36 cm away and independently optimized C finished 2.89 cm away. The 65° B/C pair also made similar forward progress with different tension. B averaged 6.58 N (44.7 N peak), C averaged 5.11 N (25.5 N peak). These trials support a direct conditioning effect and show how much of the closed-loop performance an ordinary velocity controller can recover. They do not establish a force-conditioning advantage.

## Horizon, failures and latency

| Horizon | Cases × modes | Settled A / B / C | Policy steps per update | Headless planning p50 / p95 / max | Over 100 ms |
|---|---:|---:|---:|---:|---:|
| 0.5 s | 3 × 3 development | 1/3, 2/3, 2/3 | 600 | 34.8 / 44.4 / 64.2 ms | 0/513 |
| 1.0 s | 5 × 3 | 4/5, 4/5, 4/5 | 1,200 | 65.0 / 77.4 / 100.6 ms | 1/849 |

The longer horizon settled all nine development episodes. All three shorter-horizon diagonal controllers stopped after overshooting. At A's stopped state its original 24 candidates all predicted direct robot–box contact. A fresh, wider batch yielded 15 legal candidates, but its best terminal error still matched the current 10.2 cm. The longer forecast and correspondingly longer middle-knot ramp let the same budget rank different turning and braking trajectories. The experiment changes both forecast length and the timing of the middle knot. Planning times include the snapshot, sampling, inference, physics, cost, CEM update and winner selection. Live stepping and display run separately. Validity was 12,009/12,312 at 0.5 s and 17,804/20,376 at 1 s. Poses, velocities, actions, cable vectors, crossing flags and all four substep tensions agreed exactly for all 6,785 executed policy ticks.

On the 65° start A repositioned while the box remained stationary. At the final update all 24 candidates predicted that the loaded cable would cross a geom. From the exact same state a fresh broad batch produced five valid candidates. The best predicted 74.5 cm remaining from an initial 80 cm. Repeating A from reset with optimizer seed 43 also stopped before towing. B and C transported the box but missed the settling criterion. Removing directional guidance in A improved the development diagonal (2.02 cm, 5.5 s) and allowed some transport on the 65° case (50.44 cm final error). It did not solve that case. These probes identify coupled sampling and cable-routing limitations. The next experiment should test a few broad, slew-feasible velocity proposals within the same 24 slots and check closed-loop completion on fresh steep bearings. [Trials and seeds](../tests/tether_trials.py) · [Exact-state probes](../tests/check_tether.py)

The live viewer completed the same diagonal pull in 6.3 simulated seconds. Its final rendering settings gave planning p50/p95/max of 95.4/106.4/116.0 ms, with 17 of 63 updates over 100 ms on this eight-logical-CPU host. Disabling shadows and reflections reduced misses from 48/63 in the earlier graphical run. Headless and interactive executions produced identical poses, velocities, commands and tensions. A separate two-episode window check exercised automatic reset and shutdown. Use the headless results for the native planning budget and the graphical measurements for interactive playback.

## Reproduce

From the repository root with the documented Spot assets and ONNX export available:

```bash
pixi run --locked view-tether
pixi run --locked view-tether --case diagonal --seed 17 --episodes 1
pixi run --locked env PYTHONPATH=build OPENBLAS_NUM_THREADS=1 python tools/tether_mpc.py --case side --mode matched --seed 67
pixi run --locked env PYTHONPATH=build:tools OPENBLAS_NUM_THREADS=1 python tests/tether_trials.py --horizon 1 straight diagonal heavy reverse_diagonal side
```

The viewer cycles the five initial conditions, increments the optimizer seed, reuses the ONNX session and rollout threads, and starts each episode with fresh physics and controller history. The gold cable joins the physical attachment sites, green marks the box goal, cyan shows the winning predicted box path, and the HUD reports measured tension. Use `--goal X Y` for an absolute destination, `--horizon .5` for the short reference, and `--out DIR` to retain every executed trace. The default viewer logs outcomes in `artifacts/live_tether/episodes.jsonl` and keeps the latest trace per case and mode. Box, cone and wall regression checks passed, and the refactored rate planner reproduced the archived 250-tick conditioned 15 N loaded walk exactly.
