# Uprighting a fallen cone with UniFP

The fixed demonstration starts with a cone plate 1.15 m ahead of Spot, its tip pointing toward the robot. After a one-second neutral warmup the cone rests at 102.45° from vertical. MPC first makes gripper contact at 1.44 s of control, reaches SUMO’s orientation threshold at 4.64 s and finishes the released, ground-supported hold at 6.2 s. [Executed episode with winning predictions](../artifacts/cone/selected/0/episode.mp4) · [Measurements](../artifacts/cone/selected/0/summary.json)

## Model and planning

The scene builder attaches SUMO’s traffic cone MJCF and visual OBJ from the local checkout at revision `4cea4e4e` to the existing calibrated Spot. The free body has mass 3.36 kg, principal inertias (0.25, 0.25, 0.09) kg·m² and its center of mass 0.24 m along local +Z from the plate origin. Its square plate, four feet and four stacked cylinders carry the original collision parameters. Sliding friction is 1.5 on the plate and feet and 0.5 on the cylinders. Their priority of six selects the cone’s parameters against both floor and hand. The compiled collision and inertia fields were checked against the source model and the actual contact friction was checked in the executed trajectory. In three-second passive tests with the robot out of reach, fallen and upright spawns settled at 102.45° and 0.00°, each with 32.96 N of vertical ground reaction. The cone XML carries Boston Dynamics AI Institute’s 2025 copyright notice. Source and mesh are loaded in place from the MIT-licensed SUMO repository. Generated MJBs with robot assets stay local. [SUMO object](../../sumo/sumo/models/xml/objects/traffic_cone/traffic_cone.xml) · [SUMO license](../../sumo/LICENSE) · [Scene builder](../tools/prepare_scene.py)

UniFP receives a world EE target and body vx, vy and yaw rate. The shared planner branches from the executed snapshot, samples three linear knots, evaluates native MuJoCo/ONNX trajectories, selects a valid winner, executes five 50 Hz ticks and shifts the warm start by 0.1 s. All force-command channels are zero. The native change selects the manipulated body for contact reporting. The cone trace contains signed world-frame hand and ground reactions, nongripper impulse and a flag for any active hand contact during the four physics substeps. The gripper includes the finger and terminal wrist. [Planner](../tools/box_mpc.py) · [Contact sampling](../native/world.h)

SUMO averages signed cone-axis error and gripper-to-object distance with weights 100 and 0.5. The initial UniFP comparison uses the same weight ratio and the finger-body origin reported by the rollout. For each predicted tick let `a = (R_cone e_z) · e_z`, `d` be the 3D finger-to-cone-COM distance in metres, and `f` the fraction of substeps with net hand-on-cone force above 1 N. The selected objective minimizes the horizon mean of `1 − a + 0.005 g`. Below `a = 0.9`, use `g = d` to rank approaches even before contact. At or above 0.9 use `g = 0.30 f` to encourage release. The 0.30 m equivalent scale keeps that term comparable to near-object proximity. Native failures and candidates with nongripper robot–cone impulse are ineligible. CEM selects the contact point and motion. [SUMO task](../../sumo/sumo/tasks/spot/spot_cone_upright.py) · [Reward functions](../../sumo/sumo/tasks/spot/spot_upright.py) · [UniFP cone objective](../tools/cone_mpc.py)

SUMO success is recorded at the first policy tick with `a ≥ 0.9`. Our episode ends at a prefix boundary after 25 consecutive ticks within 10° of vertical, with angular speed ≤0.25 rad/s, vertical ground reaction at least half the cone’s weight, base height ≥0.35 m and absolute robot roll and pitch below 30°. There must be no active finger or terminal-wrist contact in any of the 100 physics substeps of that hold. Orientation is invariant to quaternion sign and yaw, and distinguishes upside-down from upright. Translation is measured independently.

## Closed-loop comparisons

All rows use one CEM iteration, an independently seeded optimizer (17 here), eight CPU workers and a 12 s control limit. The SUMO-style sampler combines a 3.5 knot-noise ramp with three elites. The standard sampler uses two elites. Times below pool complete planning updates on a Xeon w3-2435, including rejected final batches. Release eligibility on the comparison traces was verified by native replay down to individual physics substeps.

| Horizon, candidates, sampler | Approach once upright | Scene seeds | Reached dot ≥0.9 | Settled and released | Plan median / p95 / max, ms | >100 ms |
|---|---|---|---:|---:|---:|---:|
| 0.5 s, 24, standard | Release | 0 | 0/1 | 0/1 | 38.8 / 46.3 / 47.7 | 0/41 |
| 1 s, 24, standard | Release | 0 | 0/1 | 0/1 | 75.5 / 88.5 / 104.2 | 1/28 |
| 0.5 s, 24, ramp | Continuous proximity | 0–3 | 0/4 | 0/4 | 37.9 / 45.9 / 49.9 | 0/332 |
| 1 s, 24, ramp | Continuous proximity | 0–3 | 2/4 | 0/4 | 75.7 / 89.6 / 124.4 | 3/397 |
| 1 s, 24, ramp | Release | 0–3 | 2/4 | 2/4 | 72.2 / 84.3 / 99.9 | 0/241 |
| 1 s, 16, ramp | Continuous proximity | 0–3 | 4/4 | 0/4 | 52.6 / 63.2 / 78.4 | 0/480 |
| **1 s, 16, ramp** | **Release** | **0–3** | **4/4** | **4/4** | **50.5 / 60.3 / 68.6** | **0/210** |

The initial 0.5 s batch on scene 0 had zero predicted gripper contacts and equal cone angles across all 24 candidates. After 4 s of execution the cone was still at 99.57° and the entire batch was ineligible. Extending the standard horizon to 1 s raised the cone to 67.34° at its best. With the ramp, 1 s forecasts discovered motions that carried it beyond the tipping point. Continuous proximity then kept the hand close enough to interfere with settling or release. For example, at the 12 s timeout on seed 0 with 16 candidates, the cone was only 0.14° off vertical but hand reaction was 22.4 N. The release objective allowed the controller to finish. The 16-candidate setting uses 800 planned policy calls per update and 3,200 physics steps. The four development starts establish a practical default for this scene family.

## Fresh starts and failure diagnosis

For nonzero seeds we vary the cone plate's forward placement from 1.10 to 1.30 m, lateral offset by ±0.10 m, robot heading by ±15°, cone-axis bearing between 155° and 205° relative to the robot, and initial tilt between 83° and 100°. The plate starts at 0.28 m height and settles under gravity. The runner rejects overlapping spawns, robot contact during warmup, or a cone that has yet to settle on its side. These variations retain the tip-facing approach and the original physical model.

| Scene / CEM seed | First gripper contact | First dot ≥0.9 | End of control | Final tilt | Outcome |
|---|---:|---:|---:|---:|---|
| 0 / 17 | 1.44 s | 4.64 s | 6.2 s | <0.01° | Released upright |
| 1 / 17 | 1.70 s | 2.88 s | 4.2 s | <0.01° | Released upright |
| 2 / 17 | 1.70 s | 4.64 s | 5.9 s | <0.01° | Released upright |
| 3 / 17 | 1.54 s | 3.16 s | 4.7 s | <0.01° | Released upright |
| Fresh 4 / 29 | 1.80 s | 3.96 s | 5.5 s | <0.01° | Released upright |
| Fresh 5 / 29 | 1.96 s | — | 6.9 s | 88.31° | No valid plan |
| Fresh 6 / 29 | 1.76 s | — | 2.1 s | 101.02° | No valid plan |

The selected controller finished **five of seven**, including **one of three fresh starts**. All 357 plans together had latency 51.3 / 59.8 / 76.2 ms (median / p95 / maximum), with zero misses of 100 ms. Peak robot roll or pitch was 8.61° across the seven episodes. On the fixed demonstration, peak 20 ms averaged net gripper force was 55.0 N and the cone COM moved (+0.531, −0.063, +0.105) m. The final hold lasted 0.54 s. In its final 25 ticks, ground support ranged from 32.77 to 36.38 N and angular speed peaked at 0.0202 rad/s. Final angular speed was below 0.000001 rad/s. All seven runs had zero executed nongripper impulse.

On both failures, every final forecast completed the native physics horizon and every one predicted nongripper impulse. Reconstructing contacts identified front-leg interference on both and torso contact on seed 6. Seed 5 reached 71.44° before losing progress. The next experiment should branch from just before those states and test candidate base/arm motions that leave the front legs clear of the cone. This family of near-side starts gives no success estimate for arbitrary bearings, and the orientation objective permits substantial translation. [Failed episode](../artifacts/cone/selected/5/episode.mp4) · [Failure replay](../tests/probe_cone.py)

## Viewer and reproduction

```bash
pixi run --locked view-cone
pixi run --locked view-cone --difficulty near --objective upright_release --headless --episodes 1 --out artifacts/cone/replay
pixi run --locked view-cone --difficulty near --scenario-seed 5 --cem-seed 29 --episodes 1
pixi run --locked view-cone --difficulty general --scenario-seed 80 --cem-seed 43 --episodes 1
```

Use `--objective upright_proximity` for the continuous-proximity comparison and `--no-sumo-sampling` for the standard sampler. To run the two box references, use `pixi run --locked view-box` or `pixi run --locked view-box --horizon 1 --candidates 24 --sumo-sampling`. With local assets prepared, each cone launch builds its own matched scene. The viewer reuses the ONNX session and worker threads over successive episodes. Saved recordings include executed states, six-channel requests, full winning EE targets and predicted finger paths. Logs carry both seeds.

Two paced graphical episodes exactly reproduced the headless outcomes and reset successfully. Scene 0 presented 28.9 poses/s, with planning median / p95 / max of 70.6 / 79.8 / 85.1 ms and no missed deadline. It played 7.2 simulated seconds including warmup in 7.51 wall seconds. A restored contact snapshot matched the executed trajectory and independently ordered serial and parallel predictions exactly. The archived 8, 35 and 50 cm box demos and full-CEM general 12 box trajectory reproduced their saved non-timing arrays. [Cone physics and parity check](../tests/check_cone.py)

## Clearance-aware whole-body planning

The current viewer default, `upright_buffer`, adds one term to `upright_release`. At each predicted policy tick MuJoCo measures the closest surface distance between the collision geoms on the torso, legs and feet and the nine cone collision geoms. The native query caps this distance at 12 cm, with bounding-sphere culling before narrow-phase queries. Let `d` be its minimum over the horizon in metres. Add `0.02 × max(0, 1 − d/0.04)²` to the original cost. Four centimetres is approximately the radius of a Spot foot. The penalty vanishes beyond that distance. Nongripper impulse in **any physics substep** still rejects the entire candidate, including unwanted arm contact. All EE and planar base commands are sampled by the existing CEM. We retained the original sampler, release cost, success checks and physical model. The wider comparison, `upright_clearance`, adds the horizon mean of `0.08 × max(0, 1 − gap/0.10)²`. [Surface distance](../native/world.h) · [Costs and scenarios](../tools/cone_mpc.py)

### Failure snapshots

Replaying the archived requests on seeds 5 and 6 reproduced every executed state. Rescoring their original CEM batches with the 4 cm term changed 14 and 5 winners respectively before termination. At 6.6 s on seed 5, the closest predicted pass widened from 4.3 to 28.2 mm while both candidates ended at 102.5° tilt. At 1.5 s on seed 6, it widened from 3.0 to 21.5 mm and terminal tilt improved from 73.2° to 72.6°. The one-second forecast already contained useful distinctions. At the failed updates every original candidate incurred a nongripper impulse, first occurring 0.28–0.96 s ahead on seed 5 and 0.50–0.96 s ahead on seed 6. Four fresh diagnostic draws of 16 at the same snapshot and pre-update distribution found respectively 9/64 and 5/64 eligible forecasts, including repeated mean/incumbent proposals in the denominator. This establishes forecast feasibility at those states. The new successful episodes used the ordinary 16 candidates from reset. [Snapshot probe](../tests/probe_cone_clearance.py)

In the original seed 5 episode, the cone reached 71.4°, fell back to its resting angle, then ended at 88.3° with a 50.9 N hand reaction and angular speed 0.50 rad/s. Reconstructed first and last loaded hand contacts were on the cone plate. With the 4 cm term the base ended at world XY (0.70, 0.43) m versus (0.62, 0.03) m. Time spent within 4 cm of the legs/torso fell from 34% to 9% of executed policy ticks. The last reconstructed loaded contact was 0.37 m up the shaft and the hand subsequently released. On seed 6 the original robot was already loading the cone at 2.1 s (27.2 N) with little rotation (101.0°, 0.13 rad/s). The new run reached the mid-shaft and released. These are observed hand locations, with no contact-point target in the cost.

### Physically settled relative poses

| Tier | Plate placement relative to Spot | Cone tip direction in the horizontal plane |
|---|---|---|
| Near | Archived forward 1.10–1.30 m, lateral ±0.10 m, seed 0 fixed | 155–205° from Spot's heading, historical drop pose |
| Varied | Range 1.18–1.50 m, bearing ±28° | 95–265° relative to the outward robot-to-plate ray |
| General | Range 1.22–1.60 m, bearing ±75° | Full 360° relative to the outward ray |

The two new tiers rotate the whole arrangement through a uniformly sampled world heading. The axis tilts 102.45° from vertical, the measured passive side-rest angle. Following SUMO’s ground-clearance principle, the generator queries MuJoCo’s plane distance to each physical cone geom and translates the whole cone until its lowest surface is 5 mm above ground. The scene is checked for overlap before the one-second warmup. The runner verifies a settled fallen angle, ground support, low linear/angular speeds and no hand touch or nongripper impulse. The generator records plate range, bearing, cone bearing, floor gap and seed. All six held-out scenes passed initialization. [SUMO ground clearance](../../sumo/sumo/tasks/spot/spot_upright.py)

### Matched closed-loop results

All rows use 16 candidates, 1 iteration, 3 linear knots, 1 s horizon, 0.1 s execution prefix and 8 workers. Historical scenes 0–3 use CEM seed 17, scenes 4–6 use seed 29. The 4 cm cost was selected on these seven, then tested on new varied seeds 70–72 and general seeds 80–82 with independent CEM seed 43. Latencies pool complete headless replans including rejected final batches. Eligibility is the percentage of all sampled trajectories that passed the native validity and nongripper checks.

| Starts | Cost | SUMO / strict successes | All-invalid stops | Eligible | Plan median / p95 / max (ms) | >100 ms |
|---|---|---:|---:|---:|---:|---:|
| Near (7) | Original release | 5 / 5 | 2 | 69.6% | 48.8 / 59.8 / 65.9 | 0/357 |
| Near (7) | Running 10 cm | 4 / 4 | 1 | 78.0% | 48.2 / 59.0 / 70.8 | 0/605 |
| Near (7) | Closest 4 cm | 6 / 6 | 0 | 75.3% | 48.7 / 58.1 / 83.1 | 0/602 |
| Varied (3) | Original release | 3 / 3 | 0 | 86.2% | 49.4 / 61.4 / 82.0 | 0/184 |
| Varied (3) | Closest 4 cm | 3 / 3 | 0 | 88.1% | 47.8 / 56.3 / 74.7 | 0/155 |
| General (3) | Original release | 2 / 2 | 0 | 90.4% | 47.6 / 58.0 / 68.9 | 0/263 |
| General (3) | Closest 4 cm | 2 / 2 | 0 | 96.0% | 47.8 / 58.6 / 64.9 | 0/270 |

| 4 cm controller, scene | First SUMO upright / strict release (control s) | Final tilt | Horizontal COM travel from start |
|---|---:|---:|---:|
| Recovered 5 | 9.18 / 10.5 | <0.01° | 0.54 m |
| Recovered 6 | 7.88 / 9.2 | <0.01° | 0.48 m |
| Varied 70 / 71 / 72 | 3.66 / 5.3, 3.64 / 5.2, 3.12 / 5.0 | all <0.01° | 0.49 / 0.53 / 0.52 m |
| General 80 / 81 | 3.10 / 9.4, 3.86 / 5.6 | both <0.01° | 0.47 / 0.54 m |
| General 82 | timed out at 12.0 | 102.45° | 0.01 m |

Across all 13 starts the 4 cm controller executed zero nongripper impulse, with peak absolute roll/pitch 8.84°. All 11 successful episodes finished the strict ground-supported, contact-free hold. The historical seed 1 regressed from success to a timeout at 88.7°, with maximum commanded arm radius on 74% of ticks. Successful near episodes took longer on average with the 4 cm clearance term. The wider penalty produced two timeouts and an invalid stop. On the six held-out starts both original and selected costs released five cones. These results support the narrower term for avoiding the original collision traps. Broader success-rate gains need more independent trials.

On general 82 the tip pointed within 7.1° of the outward ray. Both controllers approached and touched the object but timed out almost fully fallen. With the 4 cm cost, 97/120 updates forecast gripper contact. Across 1,885 eligible candidates the best predicted tilt at any tick was 102.26°. This batch evidence motivates testing more varied **joint base/EE proposals at the same budget** on plate-facing starts and measuring whether any predict rotation before changing the orientation weights. [Seed 5 recovery](../artifacts/cone_clearance/near-buffer/5/episode.mp4) · [72° side approach](../artifacts/cone_clearance/general-buffer/80/episode.mp4) · [Away-pointing timeout](../artifacts/cone_clearance/general-buffer/82/episode.mp4)

The selected controller pooled 48.4 / 58.0 / 83.1 ms planning latency across 1,027 headless updates with zero misses. Two paced viewer episodes (general 80 and 81) reset automatically, reproduced the headless commands, trajectories and eligibility exactly, and presented 29.2 and 28.9 poses/s. Graphical planning measured 69.1 / 80.0 / 92.7 ms with zero misses in 150 updates. The seven original cone trajectories and the archived 8, 35 and 50 cm box trajectories and plans also reproduced their non-timing fields exactly. [Viewer log](../artifacts/cone_clearance/viewer.jsonl) · [All matched episodes](../artifacts/cone_clearance/runs.jsonl)
