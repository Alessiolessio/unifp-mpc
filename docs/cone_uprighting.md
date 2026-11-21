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
pixi run --locked view-cone --headless --episodes 1 --out artifacts/cone/replay
pixi run --locked view-cone --scenario-seed 5 --cem-seed 29 --episodes 1
```

Use `--objective upright_proximity` for the continuous-proximity comparison and `--no-sumo-sampling` for the standard sampler. To run the two box references, use `pixi run --locked view-box` or `pixi run --locked view-box --horizon 1 --candidates 24 --sumo-sampling`. With local assets prepared, each cone launch builds its own matched scene. The viewer reuses the ONNX session and worker threads over successive episodes. Saved recordings include executed states, six-channel requests, full winning EE targets and predicted finger paths. Logs carry both seeds.

Two paced graphical episodes exactly reproduced the headless outcomes and reset successfully. Scene 0 presented 28.9 poses/s, with planning median / p95 / max of 70.6 / 79.8 / 85.1 ms and no missed deadline. It played 7.2 simulated seconds including warmup in 7.51 wall seconds. A restored contact snapshot matched the executed trajectory and independently ordered serial and parallel predictions exactly. The archived 8, 35 and 50 cm box demos and full-CEM general 12 box trajectory reproduced their saved non-timing arrays. [Cone physics and parity check](../tests/check_cone.py)
