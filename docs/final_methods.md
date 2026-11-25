# Shared-fixture and measurement validation

The comparison uses the registered Spot ReLIC simulation and CEM controller in SUMO’s installed environment. For box, a local pre-compilation hook edits only the task's object fixture in **each** task construction, before Judo copies the model into its native simulation and rollout systems. The cuboid measures 0.40 × 0.40 × 0.50 m, weighs 1 kg and has principal inertias (0.0341667, 0.0341667, 0.0266667) kg m². Its hand sliding coefficient comes from the randomized scenario (object priority 6); its explicit three-dimensional box–floor contact pair has independently sampled sliding friction. The adapter checks compiled mass, inertial frame, principal inertia, colliding geom poses, sizes, priorities, friction and solver parameters of the source, simulator and prediction model. It also checks the compiled explicit pair. The cone retains the SUMO asset. Floor/foot interactions, robot collision models, integration and servos remain policy specific. [Local adapter](../tests/sumo_common.py) · [UniFP fixture](../tools/prepare_scene.py)

ReLIC's stock arm could touch a near box during the neutral warmup. Pilot validation determined a common ReLIC box warm posture within its joint command limits, applied as a reset and constant command for one second. Its registered CEM starts normally after warmup. Both systems use the exact same free-object and planar robot spawn. The manifest gives their separately settled relative geometry. Each sees the same XY box destination, taken from the **settled** UniFP position plus the generated offset. SUMO's 3D reward uses its own settled object center height. Cone evaluation measures world-up alignment, independent of initial world translation. The registered simplified box reward uses goal-position, gripper-distance and speed. Its inherited success predicate accesses a different config field. Use the common external box score. [Registered reward](../../sumo/sumo/tasks/spot/spot_box_push.py) · [Previously reproduced predicate failure](model_uncertainty.md)

The local dynamically loaded observer runs immediately after the original MuJoCo solver call made by each native ReLIC system. It reads the live contact frame and wrench, rotates the force into world coordinates and flips its sign when the object is geom 0. Positive vertical ground force supports the object. Wrist wr1 and finger are hand bodies; other robot contacts accumulate the norm of each contact force times the substep duration. Force components from the two ReLIC substeps remain in the trace. A separate counter records real integration steps in the candidate threads; cutoff-filled states contribute zero calls. Judo's Python post-step forward pass only supplies derived visualization state. UniFP collects its existing measurements over four physics substeps. For cross-system force comparisons, integrate the norm of the **tick-mean resultant**; this differs from the integral of individual contact magnitudes. Report loaded duty above 1 N and use the separate active-contact flag for cone release. [Observer](../tests/sumo_observer.c) · [External score](../tools/physical_score.py)

**Validation before benchmark.** Four box fixtures (development easy 4, easy 1201, medium 1202, oblique 1203) and four cones (near 0, varied 305, general 310 and 82) were exercised. In all six box pilot executions, the maximum difference between box momentum change and integrated contact plus gravity was below 0.0012 N·s per 20 ms tick. The box has a centered COM, making its free-joint linear velocity directly applicable. The warmup ground reaction was within 5% of weight on both fixtures. Observer-disabled replay of the six calibration trajectories reproduced every qpos and qvel bit-for-bit. Two independently initialized pilot passes gave identical actions and trajectories on their common prefixes in all twelve scene/config assignments. The final external scorer independently reproduced all 85 archived uncertainty-trial strict classifications and cone first-upright times. [Archive check](../tests/check_common_score.py) · [Pilot driver](../tests/final_pilots.py)

The native development controller settled 3/4 boxes, with one contact-clean success. It crossed the loose cone orientation threshold on 4/4 scenes, and released 1/4. Two pilot cones finished within 1° despite failing the strict hold. Incidental robot contact can occur before a legitimate release: report both strict physical success and the subset with zero unintended impulse over the trajectory and a stable base. Early incidental contact does not terminate the SUMO controller. The two scoring conventions answer different physical questions.

UniFP's same-model branching and separate-model transfer regression checks passed. In the final experiment the warmup qpos, qvel, policy actions and commands were exactly equal across treatments and optimizer seeds within each box or cone execution plant. Tether start qpos matched across treatments. On heavy cones the offline moment reconstruction reproduced every saved joint/object configuration exactly and the original hand-force signal to numerical tolerance. Informed prediction reproduced the measured 100 ms object XY and hand-force prefixes. The earlier deterministic 8, 35 and 50 cm box replays are archived separately from the new general benchmark. [Transfer checks and archived replays](model_uncertainty.md) · [Postprocessing](../tests/final_scores.py)

ReLIC predicts a two-second horizon with three knots, 24 trajectories, three elites, a 3.5 noise ramp and nominal 20 Hz updates. Its action comes from the updated mean. It commands base motion and seven arm joints; its locomotion policy drives the legs. UniFP predicts either 0.5 or 1 s, evaluates projected EE commands plus base rates, generates all 19 joint actions, and executes the best eligible candidate's prefix. Its snapshot includes actuator delays and policy history. Judo retains its own native previous-output/reset convention. Model rates are 100 and 200 Hz, respectively, with both policies evaluated at 50 Hz. Thus count-matching is a workload control, distinct from runtime matching or matching control authority. Both native and equal-work results are essential. The latency-calibrated subset has four pairs only. [Architecture and source references](final_evaluation.md)

## Reproduction

From the repository root using the existing calibrated assets and locked environments:

```sh
export OPENBLAS_NUM_THREADS=1 PYTHONPATH=tools:build
py=.pixi/envs/default/bin/python
sh tests/build_sumo_observer.sh
"$py" tests/check_common_score.py
"$py" tests/final_pilots.py
# On a fresh evaluation workspace, materialize compiled scenes from the frozen seeds.
"$py" tests/final_scenes.py
# The supplied lock was written once, before the benchmark, by final_trials.py lock.
"$py" tests/final_trials.py general
"$py" tests/final_trials.py challenge
"$py" tests/final_trials.py sumo
"$py" tests/final_policy.py
"$py" tests/final_scores.py
```

Use the published manifest and lock to verify exact inputs before resuming. A differing source revision, environment, binary or affinity requires a separately labelled reproduction. Each completed assignment has an atomic completion record; the row includes the lock hash. The same physical scene is used for both optimizer seeds. Timing covers state transfer, optimization, all native predictions, cost and selection. CSV summaries pool individual updates for latency, retain all episode endpoints, and use paired whole-scene resampling for descriptive uncertainty intervals. With six physical challenge scenes or two in a runtime subset, intervals are coarse. Repeated seeds or thousands of MPC replans do not supply extra independent physical scenes. [Protocol](frozen_protocol.md) · [Manifest](final_scenes.csv) · [Lock and checkpoint digests](final_lock.json)

The benchmark ran serially on an Intel Xeon w3-2435, affinity CPUs 0–7, with MuJoCo 3.6.0 and ONNX Runtime 1.24.2. Pilot and optional policy rows are in their own CSVs. Each final episode has its full state and planning trace under `artifacts/final/<scene>/<CEM seed>-<physics>-<arm>/`. The raw completion record there contains the lock hash. The final per-episode CSV's scene, seed, physics and arm columns identify that record. To resume an existing run, invoke the block commands directly. The scene generation step is for fresh workspaces. [Pilot measurements](final_pilots.csv) · [Final results](final_results.md)

## Secondary and policy-level evidence reused

The original compliant-wall task rewards base travel while loading the wall. The sliding variant additionally rewards tangential hand travel with the base and retention of initial heading. These are two physical placements per arm, 15 N setpoint and 45 cm commanded travel. Means include unloaded time. Finger travel is the absolute net displacement, heading is peak excursion, and base error is the final XY distance. [Wall episodes](wall_trials.csv) · [Sliding trials](wall_sliding_trials.csv)

| Wall objective / arm | Mean force / RMSE (N) | Loaded duty | Base error / finger travel (cm) | Heading (°) | Moving RMSE (N) |
|---|---:|---:|---:|---:|---:|
| Reposition / ordinary | 13.15 / 6.65 | 92.8% | 2.1 / 38.8 | 48.1 | 5.40 |
| Reposition / force | 13.75 / 6.30 | 93.7% | 1.5 / 26.8 | 26.0 | 5.96 |
| Reposition / matched | 13.44 / 6.24 | 93.8% | 1.6 / 35.8 | 36.2 | 5.43 |
| Slide / ordinary | 13.58 / 6.80 | 92.5% | 6.2 / 40.9 | 8.4 | 6.27 |
| Slide / force | 13.27 / 6.58 | 92.9% | 8.8 / 36.1 | 8.0 | 6.06 |
| Slide / matched | 12.71 / 6.87 | 92.1% | 5.0 / 43.1 | 10.9 | 6.63 |

All six sliding executions missed the prespecified 4.5 N moving-RMSE criterion. Under known torso loads, all 27 primary walks completed; held-out path RMS was 4.30/4.64/4.42 cm ordinary/conditioned/matched. Following an unexpected pulse, return to the unloaded twin took 1.46/1.36/1.36 s after removal. These results support sustained physical contact and locomotion authority. The force-input advantage depends on the comparator and outcome. [Loaded walks](base_load.md)

Fixed-command full-state forks demonstrated different first actions at equal reachable position or velocity targets. In 12 slow towing conditions the −16 N-conditioned policy tracked its loaded velocity target better than the original matched command. A +0.016 m/s ordinary velocity calibration reproduced nearly all the improvement (loaded RMS 0.04279 versus 0.04251 m/s). In 12 fixed positive box pushes, the conditioned policy pushed farther than the nominal ordinary command, and the matched-position policy pushed farther still. Standing/walking payload and unsynchronized finger-pulse experiments likewise support a policy response, with modest or absent advantage over an ordinary command shift. These physical-response studies use distinct postures and loads and are kept separate from MPC success rates. [Policy characterization](fixed_policy_loads.md) · [Calibration](tow_velocity_correction.csv)
