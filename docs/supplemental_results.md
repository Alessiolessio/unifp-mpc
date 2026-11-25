# Supplemental experiments after the frozen UniFP-MPC benchmark

All new trajectories live under `artifacts/supplemental/`. The 538 frozen episodes retain their original assignments, scores and comparison groups. Ordinary position/velocity MPC remains the default. Fixed-command tests hold the high-level command schedule while the learned policy and its actuators continue in closed loop.

## Outcomes to carry into the report

**At comparable 10 Hz runtime, ReLIC settled more boxes and ordinary UniFP released more cones.** ReLIC also produced more incidental robot–object contacts. Planning p95 on the common suite was 11% higher for ReLIC on box and 9% higher on cone. Both controllers retained their own policy, reward, command space and collision treatment. The matched pairs are repeated optimizer seeds on the same 15 or 12 scenes. [Episodes](supplement_runtime.csv) · [Pooled timing and work](supplement_runtime_groups.csv) · [Paired comparisons with every frozen arm](supplement_runtime_pairs.csv)

| Common suite | Strict / whole-episode clean | Mean final error, all runs | First upright | Plan p50 / p95 / max, ms | >100 ms |
|---|---:|---:|---:|---:|---:|
| Box, ordinary UniFP | 12 / 12 of 30 | 54.9 mm | — | 35.7 / 42.9 / 66.1 | 0/3,355 |
| Box, runtime-comparable ReLIC | 17 / 10 of 30 | 31.6 mm | — | 41.6 / 47.7 / 65.6 | 0/2,775 |
| Cone, ordinary UniFP | 18 / 18 of 24 | 19.8° | 19/24 | 50.3 / 61.1 / 174.3 | 4/1,772 |
| Cone, runtime-comparable ReLIC | 15 / 1 of 24 | 4.9° | 23/24 | 57.3 / 66.8 / 78.2 | 0/1,584 |

ReLIC gained/lost **10/5 box** and **1/4 cone** strict outcomes against ordinary UniFP. On the seven jointly solved boxes and fourteen jointly released cones it finished 2.20 s and 2.73 s earlier on average. It completed all 5,550,000 scheduled box and 3,801,600 cone policy calls, two physics steps each. Execution added 27,750 and 15,840 physics steps after warmup. Its synchronous cycle p95/max was 51.5/69.4 ms and 71.3/83.0 ms with zero cycle misses. Mean episode hand impulses were 27.8 and 157.7 N·s. Unintended contacts occurred in 11/30 and 22/24, peak robot tilt reached 11.6° and 74.2°, and two cone episodes stopped for instability. For context, registered 20 Hz ReLIC achieved 21/30 and 9/24 strict; equal-work 10 Hz ReLIC achieved 1/30 and 19/24. [Frozen comparison groups](final_groups.csv)

**Why native ReLIC missed strict cone release.** All 24 reached the native orientation threshold. Of 15 strict failures, 14 had hand contact in the last 0.5 s. For six, hand contact alone broke the final hold. Eight exceeded the angular-speed limit, two lacked continuous support, and one failed final posture and tilt checks; those counts overlap. The hand-free failure (3107/337) briefly passed the upright threshold and fell. All 15 had earlier unintended contact. Among the nine releases, eight had earlier unintended contact and one was clean throughout. A single development ablation replaced the upright proximity penalty with the shortfall to 0.30 m finger distance once alignment reached 0.9. With the same CEM seeds, near-0/17 and general-82/43 both retained their original first-upright times (1.14, 2.78 s) and released at 2.9 and 3.7 s. The registered reward timed out on both. Only near-0 was clean. This isolates a reward effect on those two calibration starts; the main comparison uses the registered SUMO reward throughout. [Hold components per episode](supplement_release.csv) · [Development ablation](supplement_release_dev.csv)

### What helped in the fixed-goal push

Every historical replay reproduced its archived endpoint error and end-settled classification exactly. The table gives **end-settled counts / mean final XY error**. B supplies +12 N at A's nominal position. C supplies its exact +60 mm position equivalent. D uses a +20 mm ordinary position correction chosen on development mean XY error. The 1.5 kg transfer used unchanged schedules, geometry, friction, goal and the same twelve fresh placements as the 1 kg follow-up. Treat the two masses as paired. [1 kg and free-space measurements](supplement_push.csv) · [1.5 kg measurements](supplement_push_heavy.csv) · [Paired coordination and outcome differences](supplement_push_pairs.csv)

| 50 mm goal, 12 placements each | A ordinary | B force | C matched | D calibrated |
|---|---:|---:|---:|---:|
| Original 1 kg, 4101–4112 | 1 / 30.2 mm | 5 / 14.8 mm | 0 / 20.9 mm | — |
| Fresh 1 kg, 5101–5112 | 1 / 26.8 mm | 3 / 18.9 mm | 0 / 25.4 mm | 0 / 19.4 mm |
| Same placements, 1.5 kg | 0 / 30.0 mm | 5 / 18.3 mm | 0 / 23.3 mm | 0 / 20.5 mm |

- **Matched target, different delivery of impulse.** In the original scenes, B/C fingers already differed by 18.0 mm RMS before either branch acquired loaded contact. At the first identical state, elbow-0 and shoulder-1 policy actions differed by 0.132 and 0.078 in absolute actor units. Across execution the base XY paths differed by 3.1 mm RMS. A/B/C advanced the finger by 103/123/141 mm; forward impulses on the box were 1.17/1.65/2.01 N·s, absolute lateral impulses 0.058/0.105/0.179 N·s, loaded durations 0.183/0.248/0.296 s, and peak XY speeds averaged 0.113/0.250/0.312 m/s. B ended with 48.3 mm mean box progress against C's 65.7 mm. Mean maximum overshoot was 5.5/15.8 mm B/C and cross-track excursion was 6.0/12.7 mm. All replay commands were inside the spherical workspace. No unintended contacts occurred.
- **Task accuracy and EE tracking diverged.** Effective EE RMS errors on the original obstructed pushes were 56.0/75.5/68.4 mm A/B/C. End errors were 32.4/49.4/41.3 mm. With the box moved out of reach, RMS errors were still 56.0/75.3/68.2 mm for the same commands. Most of the separation in this 3 s window was present even in free motion. B had better box accuracy despite larger effective-target tracking error and shorter contact than C. On selected 4101, B ended settled at 6.2 mm and C missed at 24.2 mm. On 4109 B undershot by about 20 mm; C ended only 4.1 mm away but its final speed window reached 0.046 m/s.
- **The settling test matters.** On the original placements, five C branches spent the entire final half-second inside 15 mm, and all five failed the final speed check. On fresh 1 kg placements the corresponding counts were four C and seven D branches, all failing speed. All three B branches inside the tolerance held the speed criterion. B had lower endpoint error than C on 8/12 fresh pairs at each mass and gained 3/0 and 5/0 end-settled outcomes. Against D, B had lower error on 7/12 at each mass with the same settled gains. Calibration reproduced nearly all of B's mean 1 kg error improvement; these results establish a schedule-specific settling difference. They do not establish a general inability to calibrate an ordinary reach.

**Fixed wall.** On twelve new stationary holds, A/B/C mean normal forces were 4.00/4.76/4.51 N, loaded duties 24.8/27.2/27.6%, and mean longest uninterrupted runs over 1 N were 76/80/71 ms. Across all three arms the longest run was 90 ms. B produced higher mean force than C on 11/12 pairs, but higher duty on only three (seven lower, two tied). The highest substep peak was 56.9 N, maximum tilt was 6.91° and unintended impulse was zero throughout the scored trials. All three produced intermittent loading. [Per-hold measurements](supplement_wall.csv)

### Robot and environment mismatch

Six development runs preceded 54 fresh nominal-prediction episodes and 12 informed twins. For each cell below, the order is **A ordinary / B force / C matched**. Each number uses all three physical starts, one paired optimizer seed, including invalid plans and timeouts. [72 per-episode records, prediction errors and model hashes](supplement_mismatch.csv)

| Task / execution change | Strict successes, each out of 3 | Mean final error A / B / C | 100 ms predicted hand-force RMS error A / B / C |
|---|---|---|---|
| Box, nominal | 1 / 1 / 2 | 98.7 / 52.1 / 12.4 mm | 0 / 0 / 0 N |
| Box, lower ground friction | 2 / 1 / 0 | 10.6 / 90.4 / 60.7 mm | 0.76 / 0.48 / 1.40 N |
| Box, heavier torso | 1 / 1 / 0 | 42.1 / 22.0 / 47.5 mm | 1.33 / 1.21 / 1.39 N |
| Cone, nominal | 3 / 3 / 2 | <0.01° each | 0 / 0 / 0 N |
| Cone, lower ground friction | 2 / 3 / 3 | 34.2 / <0.01 / <0.01° | 1.96 / 1.06 / 1.62 N |
| Cone, heavier torso | 2 / 1 / 3 | 23.0 / 46.9 / <0.01° | 5.38 / 3.76 / 3.15 N |

For each box perturbation B settled one case C missed, with zero reverse pairs. On crossing 5503 with heavier torso, B settled at 10.3 mm; A and C ended at 72.5 and 88.0 mm. On low-traction boxes B had the smallest hand-force forecast error and the largest final goal error. On heavier-torso cones C gained two releases over B, which stopped twice on all-invalid batches. A incurred unintended contact on cone 5512. Informed heavy-torso prediction on box 5501 converted all three failures into successes. On cone 5511 it preserved A's success, left B unsuccessful, and lost C's release. Informed 100 ms XY and force errors were zero. Fresh box/cone pooled planning p50/p95/max were 35.0/42.7/64.3 ms and 45.7/55.7/65.5 ms, zero misses over 2,970/2,192 updates. These small paired cohorts show task- and scene-specific effects in both directions.

**Effective-target parameterization.** A development-only offline feasibility screen treated the original projected request as the intended effective position and subtracted the proposed world force divided by 200. This inverse put the nominal endpoint outside the workspace on 576/1,400 box and 2,281/7,150 cone forecast ticks. Even before projection, the derived nominal schedule exceeded the existing 0.5 m/s slew on 96/1,344 box and 1,050/7,007 cone within-plan intervals. These are correlated samples of archived winners. A fair implementation needs coupled slew and full-segment reachability for both treatments. We stopped at the feasibility screen and retained the validated command implementation. [Feasibility counts and code inputs](supplement_inverse.csv)

## Configurations and splits

- **ReLIC calibration.** Box development scenes easy-4/17 and medium-1202/223 tested 20 candidates × 2 s and 40 × 1 s, p95 48.2/48.1 ms, one strict success each. The longer horizon gave one clean success. Cones near-0/17 and general-82/43 tested 24 × 2 s and 48 × 1 s, p95 67.4/70.8 ms. Both achieved zero strict releases; mean final tilts were 0.022°/0.043°. The selection rule prioritized a ±20% p95 band around 43/61 ms, then strict count, clean count, endpoint error and latency proximity. Both selected configurations used 2 s, three knots, three elites, 10 Hz replanning, 50 Hz policy and 100 Hz physics. The 54 evaluations reused the common physical suite and optimizer seeds 311/337. Benchmark outcomes supplied no selection data. [Development lock](../artifacts/supplemental/sumo/runtime_lock.json) · [Runner](../tests/supplement_sumo.py)
- **Fixed push.** After a common contact-free warmup, each restored branch used zero base rate and nominal yaw-frame hand point (0.44, 0, −0.17) m from the spherical origin. B and C had equal reachable effective targets at a common root pose. Targets in the traces use pre-action poses; tracking error also decodes the target at the actual post-action root. Forces act on the box in world coordinates. The free reference moved the box 3 m forward. The 50 mm box goal and 3 s command duration stayed fixed. Development 5001–5006 selected D by mean goal error across +20/+35/+50 mm ordinary offsets. The 1.5 kg transfer kept D fixed. [Schedules](../tests/supplement_push.py) · [Joint diagnostics](../tests/supplement_push_pairs.py)
- **Fixed wall.** Development 0–1, validation 5201–5212. Wall face x = 0.878–0.902 m and heading ±4°. A 2 s common approach preceded a 3 s hold. B ramped yaw-forward virtual force to +10 N over 1 s and C ramped ordinary reach by 50 mm. The last 1.5 s includes all unloaded substeps. [Protocol](../tests/supplement_wall.py)
- **Mismatch.** Ground sliding friction changed from 1.0 to 0.55. Ground priority selects this coefficient at Spot's feet; the explicit box–floor pair and priority-6 cone preserve object friction. Separately, increasing the 16.708 kg torso's mass and principal inertias by 20% added 3.342 kg at its original COM. The native model copy recalculates mass constants and resets solver warm starts across mismatched dynamics. Policy history, joint ordering and delay semantics are preserved. Development used easy-4/17 and near-0/43. Validation used easy/medium/crossing boxes 5501–5503 and near/varied/general cones 5511–5513, all with CEM seed 451. Box retained the 24 × 0.5 s bounded objective with goal-axis force; cone retained the 16 × 1 s clearance/release objective with vertical force. One iteration, eight workers, nominal predictors unless explicitly marked informed. [Models, checks and assignments](../tests/supplement_mismatch.py)

## Evidence and reproduction

The supplemental raw root contains `sumo/{dev,paired,release_dev}`, `push/{replay,dev,held,free,heavy_dev,heavy_held}`, `wall/<seed>` and `uncertainty/<task>/<seed>`. The mismatch folders retain compiled prediction/execution fixtures, state and contact traces, selected forecasts, and individual plan timings. The fixed-policy folders retain commands, joint actions, targets, object motion, and warmups. ReLIC stores its issued high-level commands in `action`, post-step generalized state in `qpos/qvel`, measured substep forces in `substep_forces`, and offline finger position/body velocity in `kinematics.npz` for the 54 paired episodes. Fixed commands have no MPC forecasts. Stored 50 Hz box accelerations can be obtained by differencing measured velocities. [Raw evidence](../artifacts/supplemental/) · [File checksums](../artifacts/supplemental/evidence.sha256) · [Software, CPU and checkpoint](../artifacts/supplemental/provenance.json) · [Mismatch manifest](../artifacts/supplemental/uncertainty/protocol.json)

From the repository root, use the existing calibrated environments and one simulation process at a time. For the saved collection, verify the checksums before regenerating anything. For reproduction in a separate workspace, reuse the published `runtime_lock.json` for the 54 assignments; a fresh `dev` run also measures the calibration cohort and writes its own timing-dependent selection. Preserve the published evidence when testing a different build.

```sh
export OPENBLAS_NUM_THREADS=1 PYTHONPATH=tools:build
py=.pixi/envs/default/bin/python
relic=../sumo/.pixi/envs/default/bin/python
sha256sum --quiet -c artifacts/supplemental/evidence.sha256

"$py" tests/supplement_sumo.py dev
"$py" tests/supplement_sumo.py paired
for s in near-0 general-82; do
  PYTHONDONTWRITEBYTECODE=1 LD_PRELOAD="$PWD/build/sumo_observer.so" "$relic" tests/supplement_release_pilot.py "$s"
done
PYTHONDONTWRITEBYTECODE=1 "$relic" tests/supplement_relic_ee.py
"$py" tests/supplement_scores.py
"$py" tests/supplement_push.py
"$py" tests/supplement_push.py heavy
"$py" tests/supplement_push_pairs.py
"$py" tests/supplement_wall.py
"$py" tests/supplement_inverse.py

dest=artifacts/supplemental/build
pixi run --locked cmake -S . -B "$dest" -G Ninja -DCMAKE_BUILD_TYPE=Release
pixi run --locked cmake --build "$dest" --target unifp_native
export PYTHONPATH="$dest:tools:build"
"$py" tests/check_branching.py
"$py" tests/check_model_transfer.py
"$py" tests/check_common_score.py
"$py" tests/supplement_mismatch.py
"$py" tests/check_supplement.py
```

The supplemental build passed exact branch, command-frame and two-model continuation checks. The common evaluator reproduced 85 archived classifications. Nominal easy-4 also reproduced the archived ordinary box actions, commands, poses, velocities, finger positions, gripper forces and winning requests exactly. The original native binary remains available for frozen replays. [Validation output](../artifacts/supplemental/uncertainty.log)

**Include in the comprehensive report.** Start with the [frozen results](final_results.md), [methods](final_methods.md), [protocol](frozen_protocol.md), [episode CSV](final_trials.csv) and [paired CSV](final_pairs.csv). Include the [uncertainty study](model_uncertainty.md), [cone force study](cone_force.md), [multi-seed exploration](exploration.md), [wall sliding](wall_sliding_trials.csv), [external loads](base_load.md) and [tether transport](tether.md). For policy characterization, the [fixed-load report](fixed_policy_loads.md) and [paired measurements](fixed_policy_load_trials.csv) cover positive/negative towing, disconnected twins, standing/walking payload mass, loaded and unloaded EE height, sag, torques, saturation and tilt. The +0.016 m/s [velocity calibration](tow_velocity_correction.csv) nearly reproduced loaded tracking. Upward force and its ordinary offset restored comparable absolute hand heights while the same-command payload sag persisted. The [constant-command finger pulse](fixed_finger_pulse_trials.csv) and [synchronized pulse](final_pulse.csv) retain incremental deflection, recovery and nominal tracking scores; neither established increased stiffness. Reuse the [original fixed-goal scores](final_fixed_goal.csv) alongside the exact replays here. Keep selected [heavy push examples](ee_force_box.md) and [annotated success/failure recordings](final_results.md#selected-recordings) separate from cohort counts. Existing videos suffice for the handoff.
