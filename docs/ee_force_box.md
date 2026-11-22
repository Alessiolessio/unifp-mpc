# Does the EE force input add manipulation authority?

The force-conditioned policy produced different joint actions for the same reachable reward target and the same executed history. It helped on some loaded pushes and late corrections. Across ten paired mass/start conditions, force-conditioned MPC and ordinary position MPC each precision-settled seven. The position controller with a matched target-offset variable settled six. Position-only remains the box default.

## Input and experimental controls

UniFP computes both the finger position and spherical position target relative to the origin at root position plus the yaw-rotated offset (0.35, 0, 0.12) m. In that gravity-aligned frame the training reward is twice an exponential of the negative L1 tracking error, with exponent −2 times the error divided by the tracking scale (1.0). Its target is the nominal Cartesian target plus (virtual EE command + separately scheduled EE disturbance) / 200 N/m. With zero scheduled disturbance, the world reward target is therefore the **projected nominal world position** plus the yaw-rotated virtual force divided by 200. Ordinary contact with the box is a consequence of the dynamics. It does not enter the scheduled disturbance term. [EE geometry and force streams](../../unifp-spot/unifp_spot/envs/unifp_env.py) · [Reward](../../unifp-spot/unifp_spot/envs/base/mdp/rewards.py) · [Spot configuration](../../unifp-spot/unifp_spot/envs/spot/spot_pos_force_cfg.py)

EE force occupies commands 6–8 and actor frame entries 73–75 (zero-based), in yaw-frame newtons scaled by 0.01. Training permits ±40 N per axis. Base force occupies commands 9–11, scales by 0.01 and permits ±30 N per axis. Its horizontal components and the external base disturbance modify a linear velocity target through a damping gain of 200 N s/m and a low-speed deadband. The present experiments leave those channels and vertical EE force at zero. Only the independent external disturbance stream in Isaac applies a physical wrench. The adaptation estimator was not used to measure force. [Observation layout](../../unifp-spot/unifp_spot/envs/base/mdp/observations.py) · [Native policy step and contact-pair measurement](../native/world.h)

A, B and C share the bounded directional objective, optimized world EE position and all three optimized body rates. B and C add **one signed scalar**, along the fixed world start-to-goal line. It ramps from the previous request to one sampled value at the middle knot and holds through the final knot. Its limits are ±40 N and 80 N/s before projection. The winner, mean, interpolation and 100 ms warm-start shift use the same machinery as the base rates. The force noise uses an independent seeded stream so drawing it preserves identical underlying EE/base noise across treatments. Each planner scores 24 candidates × 25 policy ticks, fits two elites once and executes five ticks. There are eight native workers, three linear knots, 50 Hz policy and 200 Hz physics. [CEM](../tools/cem.py) · [Planner](../tools/box_mpc.py)

At each predicted or live pre-action pose, the native controller rotates the world force into the current yaw frame. It projects the nominal XYZ first. At the resulting height, allowable horizontal positions form a half-annulus constrained by radius, elevation and azimuth. The controller shortens the requested force along its ray at the first workspace boundary. B supplies the shortened force to the actor. C issues zero force and encodes exactly that reachable displaced target as a spherical position. This controls both effective reach and optimization dimension for B versus C. It deliberately excludes the extra range accessible through out-of-workspace reward targets in unconstrained force mode. Across B episodes, 0–46% of ticks required shortening by more than 0.01 N. Crossing a moving workspace boundary can change the issued force faster than the requested ramp. The largest observed world-axis change was 6.26 N in a 20 ms tick on general 62. [Projection](../native/world.h) · [Selected schedules and counterfactual checks](../tests/measure_force.py)

## Executed box results

Each cell gives final XY error in mm and elapsed control seconds. A dagger marks a timeout. All successes met the archived 15 mm, 0.5 s hold and final-prefix 2 cm/s speed criterion. Model mass and cuboid inertia scaled together. Spawn, geometry, friction, goal, and CEM seed were matched within each row. Warmup trajectories were identical between controllers. Mass changes can change physical settling during warmup. Development starts were easy 4 and general 12. General 62 and easy 17 were reserved for comparison. Easy 31 was a fresh confirmation after inspecting the 5 kg results. Conditions repeat geometries across masses.

| Mass | Scenario / CEM seed | Position A | Force B | Matched position C |
|---|---|---:|---:|---:|
| 1 kg | Easy 4 / 17 | 8.8 / 4.7 | 3.6 / 3.3 | 13.8 / 4.9 |
| 1 kg | General 12 / 17 | 9.1 / 14.9 | 25.9 / 20† | 36.2 / 20† |
| 1 kg | General 62 / 43 | 367.0 / 20† | 32.8 / 20† | 53.7 / 20† |
| 1 kg | Easy 17 / 43 | 9.6 / 8.7 | 5.2 / 2.3 | 13.1 / 7.2 |
| 1 kg | Easy 31 / 67 | 3.7 / 2.4 | 0.9 / 2.0 | 8.2 / 2.1 |
| 3 kg | Easy 4 / 17 | 17.8 / 9† | 9.5 / 1.8 | 10.8 / 5.1 |
| 3 kg | Easy 17 / 43 | 7.5 / 4.4 | 15.4 / 9† | 20.3 / 9† |
| 5 kg | Easy 4 / 17 | 16.0 / 9† | 2.1 / 2.0 | 24.2 / 9† |
| 5 kg | Easy 17 / 43 | 14.0 / 4.0 | 13.1 / 3.7 | 4.0 / 6.5 |
| 5 kg | Easy 31 / 67 | 2.5 / 2.1 | 2.8 / 2.3 | 3.9 / 2.0 |

Requested displacements were 10.4, 65.9, 63.1, 16.6 and 10.5 cm on easy 4, general 12, general 62, easy 17 and easy 31 respectively. On general 12, B made 101.7% downrange progress but finished 23.4 mm laterally off goal. On general 62 it reduced final lateral error magnitude from A's 162.4 mm to 32.2 mm and reached the coarse criterion at 17.4 s. C also substantially improved this case. On the six held-out mass/start conditions, strict counts were 5/6, 4/6 and 4/6. Across all ten, coarse counts were 9, 10 and 9, and median final errors including timeouts were 9.3, 7.4 and 13.5 mm. The samples support task-specific benefits with considerable sensitivity to the starting geometry and optimizer seed. [Per-episode contact, settling, progress, tracking, stability and latency measurements](force_box_trials.csv)

On easy 4 at 5 kg, mean resultant gripper force during loaded policy ticks was 22.8 / 31.9 / 20.3 N for A/B/C. Peak tick-mean resultants were 49.4 / 61.2 / 50.5 N. Signed gripper impulse **on the box along the goal line**, integrated over the entire episode, was 29.6 / 20.5 / 34.0 N s. B averaged +23.9 N commanded versus +20.9 N measured downrange on loaded ticks. On general 62, it averaged **−16.5 N commanded** while still applying **+3.4 N measured**. Across B episodes mean effective-target-to-finger tracking error ranged from 42 to 97 mm. Gripper and ground impulses plus gravity closed box momentum balance within 6.6e−16 N s per tick/component across all 30 episodes. Executed nongripper impulse was zero. Contact duty uses the fraction of 5 ms substeps with resultant gripper load above 1 N. Force vectors are 20 ms means. [5 kg successful force push](../artifacts/force_box/easy4-5kg/conditioned/episode.mp4) · [66 cm oblique push with residual lateral error](../artifacts/force_box/general12-1kg/conditioned/episode.mp4)

## Replayed winners

For every executed B tick we restored the identical full state, issued C's matched command for one step, restored again, then continued B. The greatest difference in world reward targets before the action was **0.098 micrometres**. On 2,683 ticks with more than 1 N supplied, the per-episode median of the largest absolute joint-action difference ranged from 0.080 to 0.348 policy units. This isolates a real input-path effect before the physics trajectories can diverge.

From each common warmup we also played the complete winning base and EE/force requests through C with the offset recomputed at each new pose. Separately we replayed B's recorded world reward targets through ordinary six-column position control with its recorded base commands. On easy 4 at 5 kg the two open-loop replays missed by 103.65 and 103.67 mm respectively, versus 2.13 mm for B. The recorded-target replay had no clipping above 10 μm. On general 12, diverging base poses caused clipping on 43.1% of recorded-target replay ticks. Replay tests identify a behavioral difference. Independent replanning gives the relevant task-performance comparison above. [All replays](../artifacts/force_box/replays.csv)

## Fixed-contact check

A fixed box presents a vertical face at world x = 0.89 m. Spot approached once, then each treatment restored that snapshot, held nominal yaw-frame (0.42, 0, 0.01) m, ramped virtual yaw-x force or its reachable position offset for one second and held for two seconds. The final 1.5 s includes zero-load samples. Surface normals were checked on the executed loaded poses. The last column is mean world finger X relative to the common branch point.

| Input | Mean wall-normal load, N | Normal impulse, N s | Finger Δx, mm |
|---|---:|---:|---:|
| Position only | 4.05 | 6.07 | −5.1 |
| +10 N | 4.57 | 6.86 | −9.2 |
| Matched +5 cm | 4.42 | 6.64 | −9.7 |
| +20 N | 4.85 | 7.28 | −13.6 |
| Matched +10 cm | 5.19 | 7.79 | −13.4 |

Contact was intermittent. Mean normal load increased modestly with requested level in this geometry. At +20 N the equivalent position offset gave greater mean load. In the archived three-seed Isaac wall protocol, baseline / +10 N / matched / +20 N / matched mean loads were 1.60 / 3.09 / 3.10 / 3.60 / 3.01 N. The protocols have different placements and trajectories. The earlier paired free-space +20 N / +10 cm native inputs moved the finger 3.51 / 5.65 cm relative to standing. Neither test establishes calibrated contact-force regulation. [Fixed-contact diagnostic](../tests/fixed_force.py) · [Isaac measurements](../../unifp-spot/docs/policy_validation.md)

## Computation and next experiment

| Controller | Plans | Full planning p50 / p95 / max, ms | Over 100 ms | Eligible / evaluated |
|---|---:|---:|---:|---:|
| A | 792 | 36.5 / 50.6 / 84.5 | 0 | 18,996 / 19,008 |
| B | 664 | 36.6 / 44.2 / 58.3 | 0 | 15,936 / 15,936 |
| C | 858 | 35.8 / 43.3 / 61.1 | 0 | 20,586 / 20,592 |

The samples above cover different executed trajectories and durations. For a paired timing check, 20 fresh plans per treatment at the same first-contact snapshot (easy 4, 5 kg), rotating measurement order and sharing seeds and workers, took median 39.6 / 41.2 / 43.7 ms. Every executed prefix across the 30 episodes exactly matched forecast poses, actions, policy inputs and gripper loads. Maximum base roll/pitch was 9.59°. The archived 8, 35 and 50 cm runs reproduced every non-timing episode and plan array, and general 12 again reached 9.05 mm with A. Native branching and cone contact regression checks passed.

There was no specified desired physical load for box translation, and total impulse did not consistently predict success. The box cost therefore stays unchanged. For the next milestone, prescribe a *measured* wall-normal load and compare the same three planners on settling time, load error including gaps, and peak loads. Establish nominal tracking before adding identical external disturbance pulses.

## Reproduction

From the repository root with the runtime and assets prepared:

```bash
pixi run --locked view-box --difficulty easy --scenario-seed 4 --cem-seed 17 --box-mass 5 --ee-force conditioned --episodes 1
pixi run --locked env OPENBLAS_NUM_THREADS=1 PYTHONPATH=tools:build python tests/force_box.py --mass 1 --cases 0 1 2 3 4
pixi run --locked env OPENBLAS_NUM_THREADS=1 PYTHONPATH=tools:build python tests/force_box.py --mass 3 --cases 0 3
pixi run --locked env OPENBLAS_NUM_THREADS=1 PYTHONPATH=tools:build python tests/force_box.py --mass 5 --cases 0 3 4
pixi run --locked env OPENBLAS_NUM_THREADS=1 PYTHONPATH=tools:build python tests/measure_force.py
pixi run --locked env OPENBLAS_NUM_THREADS=1 PYTHONPATH=tools:build python tests/fixed_force.py
```

Use `--ee-force matched` for C and `--ee-force off` for A in the box viewer or box MPC CLI. The violet viewer line shows the issued spherical **position** decoded in world coordinates. In B, the training reward target additionally includes the issued force offset. All native inputs to the policy and the requested world force vectors are saved in headless episode and plan traces.
