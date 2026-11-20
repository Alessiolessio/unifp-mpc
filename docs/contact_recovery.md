# Box contact recovery

Run the paced side-approach demonstration from the repository root. Omit `--episodes 1` to continue with new oblique scenes. The usual `view-box` command still cycles through the original tiers with forward-only MPC.

```sh
pixi run --locked view-box --difficulty oblique --scenario-seed 0 --cem-seed 17 --base stance --episodes 1
```

## What changes in stance mode

CEM still samples world EE XYZ and body-forward speed at three knots. Once per planning update, the controller computes two supporting commands from the executed base pose, box center and existing local push waypoint. The desired lateral speed is 1.5 times the body-left displacement to a stance one metre behind the box along the desired push. The yaw rate is 1.5 times the signed bearing from the base to the box. Inside 15 mm of the local waypoint, dividing the push vector by 15 mm attenuates the stance offset. All candidates share these two profiles during the update. The optimizer can still change the approach with its finger target and forward/backward speed.

The trained Spot command configuration specifies ±0.6 m/s forward, ±0.4 m/s lateral and ±0.6 rad/s yaw. MPC uses ±0.35, ±0.22 and ±0.50 respectively. The planner limits each linear axis to 0.8 m/s² and yaw to 0.8 rad/s². Each rate starts at the last executed request, ramps to knot 1, then holds. Native commands contain world XYZ followed by body vx, vy and yaw rate. Projection into the shoulder workspace uses each rollout's pre-action base pose at each 50 Hz policy tick. Force inputs stay zero. The viewer draws the same selected EE requests and predicted finger samples used in the headless controller. [Spot command configuration](../../unifp-spot/unifp_spot/envs/base/base_cfg.py) · [Sampling and guidance](../tools/box_mpc.py) · [Native mapping](../native/world.h)

The baseline, ablations and stance controller all use the Judo-derived two-elite update, 24 candidates, one iteration, a 0.5 s horizon, a 0.1 s executed prefix and eight workers. The bounded directional objective and settling rule are common to all primary comparisons. The `oblique` tier rotates a 1.30–1.50 m approach by a signed 30–50°, adds up to 5 cm of transverse offset, and asks for a 24–42 cm push within ±15° of that rotated axis. Spawn collision checks and warmup checks still run. [Judo CEM](../../judo/judo/optimizers/cem.py) · [Scenario generator](../tools/box_scenario.py)

## What happened at contact

On hard 9, the reference box reached the requested downrange distance at 5.5 s with 59 mm of transverse error. The final corrective direction then pointed 120° from the base heading. The radial request stayed at the policy's 0.55 m upper bound from 5.5 to 9 s. At 5.5 s all 24 candidates made measured contact, yet even the most favorable valid terminal state increased final-goal error by 2.60 mm. The selected rollout's one reconstructed tick-end contact point was high on the negative-X box face, at box-local (−0.200, −0.031, +0.216) m, with an inward-normal dot product of −0.067 against the required XY direction. Its net gripper impulse along the correction was just +0.034 N s. At 8 s no candidate reached the box.

Ranking mattered earlier. At 4 s the directional winner predicted 60.1 mm of final-goal progress, a SUMO-inspired center-proximity rerank of the same batch predicted 65.4 mm, and the largest available terminal progress was 79.6 mm. The optimizer also scores running error and the intermediate waypoint. Switching the forward planner to proximity at the 4.0 or 5.5 s cut finished at 61.9 or 61.8 mm. Both directional continuations finished at 19.3 mm. At 5.5 s switching cost could not supply a useful missing action. Normals and locations here come from MuJoCo contact reconstruction at recorded tick-end poses. Signed impulses integrate the native gripper–box forces across all four physics substeps. [Contact diagnostic](../tests/inspect_recovery.py) · [SUMO box objective](../../sumo/sumo/tasks/spot/spot_box_push.py)

Long 14 reached 123 mm cross-track error by 6.5 s. All 24 candidates touched in the next forecast, with only 0.17 mm of terminal progress available. Later the gripper contacted a different corner. The mean inward-normal alignment at sampled contact points rose from 0.12 in the 5.5–6.5 s window to 0.86 in the 6.5–8 s window. These observations implicate contact geometry and the available short-horizon motions. The policy's mechanical reachability limit remains unmeasured.

## Restored-state comparisons

Each cut replayed the archived forward controller, then restored exactly the same native snapshot before each continuation. The shared CEM coordinates, incumbent, fitted variance, previous request and random-generator state were copied. Newly enabled sampled axes started at zero with the initial variance. Additional axes consume additional random draws after branching. Every forward reference continuation reproduced the archived qpos exactly. All controllers used the time remaining until the original 9 s deadline.

| Snapshot after warmup | vx | vx + vy | vx + yaw | vx + vy + yaw | Guided stance |
|---|---:|---:|---:|---:|---:|
| Hard 9, 4.0 s | 19.3 | 22.5 | 24.5 | 39.5 | 68.6 |
| Hard 9, 5.5 s | 19.3 | 49.6 | 44.6 | 54.9 | 87.1 |
| Long 14, 4.0 s | 50.7 | 26.7 | 73.6 | 127.2 | **4.6 settled** |
| Long 14, 6.5 s | 50.7 | 36.4 | 41.8 | 47.8 | 13.7, hold incomplete |

Values are final XY error in mm. On the long 14 early branch, guided stance recorded 97 corrective contact ticks versus 61 for forward-only, and ended 0.03 mm off the push line versus 49.0 mm. A corrective tick has measured contact, positive gripper impulse toward the final goal and decreasing box-goal distance. On the late branch the first corrective tick arrived after 0.46 s versus 0.94 s. In a separate paired 12 s episode-limit check, stance settled at 9.4 s with 12.8 mm error. Forward remained 50.7 mm away. Neither hard 9 cut settled with either controller by 12 s. The cuts within each episode are correlated.

We also tried steering only toward the box, centering it with lateral speed alone, and splitting proposals among straight, turn and sidestep. Lateral centering alone settled the late long 14 cut in 2.2 s, but finished 87.6 mm away from the earlier cut and missed both hard 9 cuts. Replacing working distance with a neutral arm-reach penalty caused some batches to run out of valid plans. Reopening CEM variance after a batch predicted under 1 mm progress produced no additional settled branches. A lateral-workspace trigger that delayed stance activation until 2.3 s on hard 9 and 3.3 s on long 14 also missed both from reset. These checks support keeping stance opt-in. [Paired recovery runner](../tests/recover_box.py)

## Episodes from reset

Success requires 25 consecutive policy ticks within 15 mm and XY speed at most 0.02 m/s over the last five ticks. All six paired starts used matching live and prediction physics and had zero executed nongripper impulse. Peak absolute roll/pitch for guided stance was below 9.6°.

| Scene / CEM seed | Forward final error | Stance final error | Stance control time | Stance contact ticks |
|---|---:|---:|---:|---:|
| Hard 9 / 17 | 19.3 mm | **9.9 mm settled** | 6.6 s | 63 |
| Long 14 / 29 | 50.7 mm | 100.8 mm | 9.0 s | 153 |
| Oblique 0 / 17 | 408.3 mm | **3.1 mm settled** | 5.5 s | 112 |
| Oblique 1 / 17 | 245.0 mm | 83.1 mm | 3.9 s, no valid plan | 41 |
| Held-out oblique 2 / 29 | 358.3 mm | 26.7 mm | 9.0 s | 82 |
| Held-out oblique 3 / 29 | 260.5 mm | 22.6 mm | 9.0 s | 58 |

On oblique 0 the robot turned 41.3° before first gripper contact at 2.44 s. The baseline never touched the box on any of the four oblique starts. Replaying the successful commands on seed 0 with yaw zeroed yielded zero contact ticks and 408.3 mm error. Zeroing lateral speed alone left 35.1 mm error. Keeping the base requests but holding the warmup arm command produced zero gripper contact and 1.81 N s of nongripper impulse. These are open-loop channel interventions on the winning schedule. The table above uses independently replanned closed loops. On hard 9 the successful run still made a 78.2 mm peak lateral excursion, compared with 67.9 mm for the reference. The improvement there was in finishing the push.

## Latency and verification

| Workload | Full planning p50 | p95 | Maximum | 100 ms misses |
|---|---:|---:|---:|---:|
| Six stance episodes headless, pooled | 36.4 ms | 43.0 ms | 53.9 ms | 0/431 |
| Oblique 0 through the viewer entry point, headless | 38.3 ms | 45.0 ms | 46.1 ms | 0/55 |
| Same episode, paced interactive window | 53.1 ms | 63.7 ms | 64.8 ms | 0/55 |

The window presented 191 executed poses at 29.0 pose frames/s. The 6.5 simulated seconds including warmup took 6.75 wall seconds including episode setup. All 55 selected prefixes matched their forecasts exactly, and the graphical and headless runs had identical contact counts and final errors. Native tests checked zero-yaw compatibility, raw policy-command replay and independent world-target projection during a sustained turn. The original 8, 35 and 50 cm demonstrations reproduced all archived non-timing trace and plan arrays exactly. [GUI log](../artifacts/recovery/live.jsonl) · [Headless log](../artifacts/recovery/headless.jsonl) · [Native checks](../tests/check_branching.py) · [Viewer-path checks](../tests/check_view_box.py)

To reproduce the snapshot tables with the earlier hard 9 and long 14 archives available, run the following after building the native module. On a fresh workspace, generate the hard 9 scene with `tools/box_scenario.py --seed 9 --difficulty hard` and long 14 with `--seed 14 --long-push`. Save the forward `tools/box_mpc.py --controller mobile --objective bounded --candidates 24` runs with CEM seeds 17 and 29 to the archive paths listed in the recovery runner.

```sh
export PYTHONPATH=build:tools OPENBLAS_NUM_THREADS=1
pixi run --locked python -B tests/recover_box.py
pixi run --locked python -B tests/inspect_recovery.py
pixi run --locked python -B tests/run_recovery_scene.py oblique0 oblique1
pixi run --locked python -B tests/run_recovery_scene.py oblique2 oblique3 --seed 29
```

The next experiment is to reserve a candidate that backs off before approaching an adjacent face after a batch shows contact without corrective progress. The late hard 9 snapshot and oblique 1's exhausted valid batch supply matched failure states to test that maneuver.
