# Final evaluation of UniFP-MPC

**Ordinary UniFP-MPC remains the default.** Vertical EE force gave the highest strict release count on the general cone benchmark, 20/24, against 18/24 for both ordinary and matched vertical-position control. Its advantage did not carry over to the heavier, away-pointing cones. A reproducible force-specific benefit appeared in fixed-command 1 kg pushing, where conditioning reduced error at a prespecified goal relative to both controls. Native SUMO settled more boxes and brought every general cone through its native upright threshold. UniFP achieved more sustained, gripper-free releases with a stable robot.

The evaluation comprises 282 general UniFP episodes, 144 paired blind/informed challenge episodes, and 112 SUMO episodes. All 51 physical starts passed initialization. All 538 assignments, including timeouts, invalid-plan stops, unwanted contact and falls, enter the results. Each principal comparison pairs the same physical scene with optimizer seeds 311 and 337. The box set retains the original four tiers and adds three general-tier scenes, a decision made before final execution. No controllers were retuned on these outcomes. [Frozen protocol](frozen_protocol.md) · [Scene manifest](final_scenes.csv) · [Per-episode data](final_trials.csv) · [Group statistics](final_groups.csv) · [Paired differences and uncertainty](final_pairs.csv)

## General held-out manipulation

The ordinary controller optimizes position or velocity. The force controller also supplies a virtual force input to the policy. Its matched control uses the same number of search variables and feasible target offsets, then issues an ordinary position or velocity command. Each controller replans independently. Error below is the mean over **all** endpoints. Strict success requires 0.5 s of settling. Cone release also requires adequate ground support, low angular speed, a stable robot and absence of hand contact throughout the hold.

| Task | Optimized force or equivalent offset | Ordinary strict / error | Force strict / error | Matched strict / error |
|---|---|---:|---:|---:|
| Box, 15 scenes × 2 | Goal-axis EE | 12/30 · 54.9 mm | 11/30 · 39.9 mm | 10/30 · 46.4 mm |
| Cone, 12 × 2 | Vertical EE | 18/24 · 19.8° | 20/24 · 15.4° | 18/24 · 22.8° |
| Same cones | XYZ EE | 18/24 · 19.8° | 16/24 · 22.3° | 19/24 · 19.3° |
| Tether, 12 × 2 | Horizontal XY base | 17/24 · 37.7 mm | 17/24 · 47.2 mm | 18/24 · 36.6 mm |

**Box.** The box began 1.04–2.23 m from the robot, with requested displacements of 0.125–0.575 m. Easy scenes settled in 5/6, 6/6 and 4/6 ordinary/force/matched runs. The crossing tier yielded 1/6, 0/6 and 0/6. The added general tier yielded 2/6, 1/6 and 2/6. Force gained three and lost two strict outcomes against matched position. On their eight jointly settled pairs force took 0.99 s longer on average. Mean maximum overshoot was 20/19/23 mm, endpoint undershoot 18/3/14 mm and maximum cross-track error 66/70/71 mm. Executed paths had zero unintended robot–object impulse. The lower average error for force is evidence of better endpoints on this sample, alongside a lower settling count than ordinary MPC.

**Cone.** Ordinary, vertical force, matched vertical, XYZ force and matched XYZ crossed the native axis threshold in 19/20/19/17/20 of 24 episodes respectively. Vertical force gained four releases and lost two against its matched control. The paired success difference was +8.3 percentage points, with a 95% scene-bootstrap interval of −12.5 to +33.3. On the 16 jointly successful pairs it took 0.38 s longer on average. XYZ force gained three and lost six against matched XYZ (−12.5 points, interval −37.5 to +12.5). Although XYZ force achieved at least 10° of post-contact progress in 23/24 runs, six ended with no valid plan, versus two for matched XYZ. More early rotation did not ensure safe completion. All strict UniFP releases also passed the whole-episode posture and unintended-contact check.

**Tether.** Ordinary and base-force MPC settled exactly the same 17 pairs. Force gained one and lost two against matched velocity. Ordinary/force/matched solved 8/8, 8/8 and 7/8 in the lowest bearing tier, and 2/8, 2/8 and 3/8 in the steepest tier. Mean physical tension was 3.47/3.46/3.54 N, maximum tension 53.3/57.0/58.3 N, and mean maximum cross-track error 75/83/76 mm. At the substep tension threshold of 0.1 N, mean slack-to-taut entries were 89/90/91 per episode, including brief unloading during gait. All six terminal invalid batches included cable-crossing rejections. The executed paths had zero direct robot–box impulse. Maximum robot roll or pitch remained below 9.1°.

The rates describe these frozen distributions. The general cones varied in approach distance and bearing, but most tips pointed back toward the robot. The targeted family below measures a different contact problem. Uncertainty intervals resample whole physical scenes with both seeds together. Per-treatment rates, paired wins and losses, joint completion times and intervals are available for all 64 contrasts. [Group statistics](final_groups.csv) · [Paired contrasts](final_pairs.csv)

## Does conditioning help when the object is heavier than predicted?

Each row below uses six fresh physical scenes × two optimizer seeds. Blind and informed trials execute the same heavy object. Only the prediction physics changes. Each cell gives strict success and mean final error including failures.

| Execution plant | Prediction mass | Ordinary | Force | Matched |
|---|---:|---:|---:|---:|
| 5 kg box | 3 kg, blind | 7/12 · 29.7 mm | 6/12 · 18.6 mm | 7/12 · 16.6 mm |
| Same boxes | 5 kg, informed | 8/12 · 22.5 mm | 7/12 · 22.5 mm | 5/12 · 25.9 mm |
| 4.20 kg away-pointing cone | 3.36 kg, blind | 0/12 · 101.4° | 0/12 · 102.4° | 1/12 · 93.9° |
| Same cones | 4.20 kg, informed | 1/12 · 93.9° | 1/12 · 93.6° | 1/12 · 93.9° |

For boxes, blind-only/informed-only successes within each treatment were 1/2 ordinary, 1/2 force and 4/2 matched. Force lost two and gained one against matched position under blind prediction, then gained two and lost none under informed prediction. At the executed 100 ms prefix, mean per-episode RMS object XY errors were 2.36/2.32/2.27 mm for blind ordinary/force/matched. The corresponding physical hand-force vector errors were 4.09/4.01/3.30 N. Both discrepancies were zero with informed models. A better model improved prediction fidelity in every arm, while discrete contact choices and closed-loop settling continued to vary. Archived common-command branches give errors at additional lookaheads. [Prediction validation](model_uncertainty.md)

For the blind cones, only 1/1/2 episodes showed at least 10° of useful tilt progress after contact. Average impulse-weighted contact locations were 2.2/2.2/2.6 cm up the cone axis. Mean signed hand angular impulse about measured ground support was −0.32/−0.34/+1.00 N·m·s despite 43.4/41.2/37.8 N·s of hand impulse. Positive angular impulse points toward upright. The force arm averaged 98% valid candidates. These observations point to how the hand loaded the plate and shaft as a limitation even when legal commands were plentiful. The blind 100 ms XY forecast errors averaged only 0.26/0.14/0.21 mm. A motionless cone is easy to predict and difficult to upright. Informed prediction added one strict release for ordinary and one for vertical force. Matched position released the same single pair in both conditions. One blind ordinary execution incurred unintended contact. With six independent geometries, zero observed successes still permits a one-sided 95% upper bound of 39% on the chance that **either** seed solves a new, similarly drawn geometry.

Two selected examples put the challenge results in context. On historical cone 82, vertical force released four of ten nominal optimizer repeats against zero for ordinary and matched vertical. Under a hidden 25% mass increase, one selected vertical-force run still released in 9.3 s. In the final blind scene 3410, seed 311, the outcome reversed. Vertical force generated +11.94 N·m·s yet ended at 102.45°. Matched vertical generated +10.54 N·m·s, contacted farther along the axis (8.3 versus 3.8 cm), and released in 6.5 s after 47.5 cm of cone translation. The historical 5 kg easy-4 box likewise precision-settled with ordinary and force control while its matched control missed. These demonstrations establish feasible behaviors. The frozen challenge families provide the estimates of recurrence. [Historical evidence](final_evaluation.md) · [Final contact and episode measurements](final_trials.csv)

## Direct SUMO/ReLIC comparison

Both systems used the same compiled object fixture, object spawn and box XY goal. The 1 kg cuboid has full dimensions 0.40 × 0.40 × 0.50 m and an explicit randomized friction pair with the floor. The actual ReLIC execution and prediction models were checked. Following warmup the largest cross-system difference in cone object XY was 1.12 mm. Each policy retained its own valid reset posture, robot model, actuator dynamics, commands and task cost. The observer sampled integrated contact at 100 Hz for ReLIC and existing instrumentation at 200 Hz for UniFP. The common external evaluator reproduced all 85 archived UniFP classifications. A box momentum check bounded the dev-pilot measurement residual to 0.0012 N·s per tick. [Fixture, observer, pilots and scoring](final_methods.md)

| Controller | Box settled / clean | Box mean XY error | Cone axis ≥0.9 | Cone released / clean | Cone mean end tilt |
|---|---:|---:|---:|---:|---:|
| Ordinary UniFP | 12 / 12 of 30 | 54.9 mm | 19/24 | 18 / 18 of 24 | 19.81° |
| Native SUMO, 20 Hz | 21 / 18 of 30 | 12.7 mm | 24/24 | 9 / 1 of 24 | 0.53° |
| SUMO, equal scheduled work, 10 Hz | 1 / 0 of 30 | 75.8 mm | 23/24 | 19 / 4 of 24 | 3.52° |
| SUMO, dev latency-calibrated subset, 10 Hz | 0 / 0 of 2 | 154.6 mm | 2/2 | 1 / 1 of 2 | <0.01° |

“Clean” additionally requires zero unintended robot–object impulse and stable posture from start through completion. Strict cone release already requires a stable, hand-free final hold. Under native SUMO seven of nine releases also had zero unintended contact during that hold. Under equal-work SUMO the count was 17 of 19. Earlier contact accounts for much of the difference from whole-episode clean success. Force traces distinguish continued loading from an upright cone left standing.

Native SUMO gained eleven box settlements and lost two against ordinary UniFP (+30 percentage points, 95% scene-bootstrap interval +6.7 to +53.3). It settled 3.21 s earlier on their ten jointly successful pairs. On cone release, ordinary UniFP gained twelve and lost three (+37.5 points, interval +8.3 to +66.7). Native SUMO released 2.75 s earlier on their six joint releases. Fourteen of its fifteen strict failures still had active hand contact in the last half-second. It accumulated 389 N·s mean hand impulse, against 83 N·s for ordinary UniFP. Equal-work SUMO released one more cone overall than ordinary UniFP, but recorded unintended contact in more episodes and three unstable terminations. The matched-work box controller had six candidates with the registered two-second horizon. Its result shows the sensitivity of this architecture to the budget and its allocation. The different rewards, safety handling, command authority and optimization settings preclude attributing cross-system differences to the force channel.

## Compute and deadlines

The following are pooled **planning** times in ms on the locked eight CPU cores. Ranges give the lowest and highest treatment statistics separately. Deadlines count real plan duration. Completion time above is simulated time. Rendering, force replay and plotting took place offline.

| Configuration | Policy evaluations per update | p50 | p95 | Maximum | Over native period | Over 100 ms |
|---|---:|---:|---:|---:|---:|---:|
| UniFP general box, 10 Hz | 600 | 35.7–35.9 | 42.9–43.1 | 54.0–93.1 | 0/10,394 | 0 |
| UniFP general cone, 10 Hz | 800 | 50.1–50.6 | 59.6–63.4 | 130.0–196.4 | 49/8,193 | 49 |
| UniFP general tether, 10 Hz | 1,200 | 68.6–69.2 | 80.2–81.3 | 90.9–96.6 | 0/4,681 | 0 |
| Native SUMO box, 20 Hz | 2,400 | 45.0 | 50.4 | 60.0 | 287/4,362 | 0 |
| Native SUMO cone, 20 Hz | 2,400 | 56.3 | 63.9 | 75.6 | 3,962/4,432 | 0 |
| Equal-work SUMO box, 10 Hz | 600 | 14.7 | 18.7 | 22.9 | 0/3,959 | 0 |
| Equal-work SUMO cone, 10 Hz | 800 | 20.0 | 25.2 | 33.5 | 0/1,659 | 0 |
| Latency-calibrated SUMO box / cone, 10 Hz | 1,600 | 31.4 / 37.1 | 35.0 / 43.8 | 38.7 / 47.3 | 0/440 | 0 |

All scheduled policy evaluations completed, including trajectories subsequently rejected on physical grounds. The blocks executed 18,408,000 general UniFP, 9,381,200 challenge UniFP and 25,512,200 SUMO policy ticks. Native SUMO scheduled 48,000 ticks per simulated second, compared with 6,000 for UniFP box and 8,000 for UniFP cone. The native SUMO cone planner exceeded its 50 ms period on 89.4% of updates. The latency-calibrated subset ran faster than UniFP on the reserved scenes and has only two cases per task. It establishes approximate planning-time comparability, with insufficient evidence for an outcome ranking. On the heavy challenges UniFP missed zero 100 ms deadlines in 12,870 updates. [Per-arm timings, maxima and actual work](final_groups.csv)

## What can be attributed to force conditioning?

A fixed, three-second command schedule on twelve fresh 1 kg box placements used a **50 mm goal chosen before evaluation**. Ordinary, +12 N virtual EE force and matched +60 mm position adjustment ended settled in 1/12, 5/12 and 0/12. Mean final XY errors were 30.2/14.8/20.9 mm. Force reduced error relative to its matched control on seven pairs and gained five settled outcomes with no losses. The ordinary command usually undershot. The larger ordinary position offset overshot more (mean maximum 15.8 mm versus 5.5 mm with force) and had more cross-track error. This identifies a physical capability at a specified goal and contact schedule. Independently replanned ordinary commands can choose other offsets. [Fixed-goal episodes](final_fixed_goal.csv)

A second twelve-condition test synchronized +2 N vertical virtual force or its +10 mm position equivalent with a 0.4 s downward physical finger disturbance. Against each treatment's own undisturbed twin, mean peak deflections were 33.39/33.45/33.63 mm ordinary/force/matched and recovery times were 0.657/0.670/0.670 s. The treatments changed nominal arm posture but conferred little advantage in incremental disturbance rejection. Previously, ordinary velocity calibration also reproduced nearly all the fixed towing advantage. In wall sliding, force maintained 92.9% loaded duty, but all six sliding trials missed the moving force-RMSE criterion. [Pulse measurements](final_pulse.csv) · [Wall, loaded locomotion and policy diagnostics](final_methods.md#secondary-and-policy-level-evidence-reused)

The evidence supports keeping vertical EE conditioning available for cone experiments and ordinary position/velocity as the general baseline. It supports a narrower claim for the TCC: the trained force input can change useful contact behavior at a matched reachable target, with outcomes depending on contact geometry and the success criterion. Generalizing from the selected heavy box or away-pointing cone would overestimate robustness. Better understanding of how often the robot acquires and releases a load-bearing shaft contact is a more specific open question than whether larger virtual forces help.

## Selected recordings

The cone pairs share their physical scene and optimizer seed. The ordinary box and tether examples show a success from each challenging tier. These are illustrative choices made after scoring. The counts above include all outcomes.

| Scene and seed | Video and telemetry |
|---|---|
| General cone 3109 / 311, vertical force, release at 9.1 s | [Video](../artifacts/final/3109/311-known-Bz/dashboard.mp4) · [Plot](../artifacts/final/3109/311-known-Bz/dashboard.png) |
| Same start, matched vertical, invalid batch at 3.6 s | [Video](../artifacts/final/3109/311-known-Cz/dashboard.mp4) · [Plot](../artifacts/final/3109/311-known-Cz/dashboard.png) |
| Heavy away cone 3410 / 311, blind matched vertical, release at 6.5 s | [Video](../artifacts/final/3410/311-blind-Cz/dashboard.mp4) · [Plot](../artifacts/final/3410/311-blind-Cz/dashboard.png) |
| Same start, blind vertical force, timeout at 12 s | [Video](../artifacts/final/3410/311-blind-Bz/dashboard.mp4) · [Plot](../artifacts/final/3410/311-blind-Bz/dashboard.png) |
| General box 3015 / 311, ordinary, settled at 13.8 s with 10.3 mm error | [Video](../artifacts/final/3015/311-known-A/dashboard.mp4) · [Plot](../artifacts/final/3015/311-known-A/dashboard.png) |
| Steep tether 3212 / 311, ordinary, settled at 5.3 s with 10.8 mm error | [Video](../artifacts/final/3212/311-known-A/dashboard.mp4) · [Plot](../artifacts/final/3212/311-known-A/dashboard.png) |
