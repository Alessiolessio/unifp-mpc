# Force-conditioned MPC with a nominal prediction model

Vertical EE conditioning preserved a useful cone-tipping behavior with a cone 25% heavier than the predictor assumed (3.36 kg predicted, 4.20 kg executed). On two configurations it produced a strict release where the reachable vertical-position control failed. Two other optimizer seeds on the away-facing cone failed with all three treatments. On three fresh combined-mismatch box starts, ordinary, conditioned and matched MPC settled the same two boxes. The conditioned controller had the largest mean final error. Ordinary control remains the default. [Individual results](uncertainty_trials.csv) · [Forecast errors](uncertainty_prediction.csv) · [Evidence inventory and final protocol](final_evaluation.md)

## Model transfer and controls

Construct `Rollouts(nominal, workers)`, then `live = pool.execution_world(mass_scale, floor_scale)`. The native factory deep-copies the **compiled** nominal MuJoCo model for execution. Only object mass and all three principal inertias at fixed geometry, and the explicit box–floor pair's two sliding coefficients, can change. Mass changes rebuild derived model constants with `mj_setConst`. The original prediction model, ONNX session and worker allocation stay in place. This constrains compatible topology by construction. Arbitrary MJB-to-MJB transfer is outside this API. The experiment saves the execution MJB for independent replay. [Native factory and import](../native/rollouts.h) · [Trial definitions](../tests/model_uncertainty.py)

Snapshots from the current registered execution clone can enter this pool. Other model identities and superseded clones are rejected. Replacing an episode revokes the clone registration. Native same-world restoration retains its original guard and implementation. Per batch, the importer transfers integration state including time, qpos, qvel, actuator state, UniFP's 32 frames, previous actions, gait tick and target-delay ring including its cursor. Identical models keep solver warm-start for bitwise continuation. Different physics start the nominal solver with zero acceleration warm-start. Past interval wrench summaries are cleared. The next step rebuilds contact constraints and dynamics on the nominal plant. External applied generalized and body wrenches are cleared by default. A caller may opt into `known_load=True` for a genuinely observed sustained load. Hidden applied loads also clear the solver seed. The planner sees the current executed state every 100 ms. Only the external evaluator reads the actual cone mass to apply the original half-weight ground-support criterion.

The layout test checks ordered names, state size, body/joint/geom/site/actuator/tendon mapping, command metadata, knee table, spawn and cable routing, plus exact modified mass, principal inertia and pair friction. Two separately allocated identical models gave **bitwise equal** actions, issued commands, poses, velocities, EE positions and contact fields at reset, partial history and during loaded box contact. Tests also rejected stale snapshots, permuted lanes, hid an external load and preserved a known load. All three easy-4 controllers reproduced their archived physical trajectories. The 8/35/50 cm demonstrations reproduced their archived non-timing episode and plan arrays, finishing 5.81/3.03/9.80 mm from goal. Same-model branching, cone rest/contact, 512 XYZ-force command projections, wall replay, base-load and tether checks passed. [Transfer check](../tests/check_model_transfer.py)

## Box uncertainty

All runs use the bounded directional objective, full EE position and three base rates, 24 × 25 policy ticks, eight workers, one CEM update and a five-tick prefix. B adds a signed goal-axis force. C samples the same scalar and encodes B's reachable offset at the current base yaw with force zero. Within a physical case all three have identical warmup poses, velocities, actor outputs and issued commands. The benchmarks use the existing 15 mm settling, 0.5 s hold and final-prefix speed criterion. Actual nongripper contact ends a mismatched run. No such contact occurred.

Final XY error in mm includes timeouts (†). A/B/C are ordinary / conditioned / matched.

| Physical condition | A | B | C |
|---|---:|---:|---:|
| Development easy 4, 1 kg nominal and actual | 8.8 | 3.6 | 13.8 |
| Same start, 0.70 kg actual | 11.8 | 5.6 | 17.4† |
| Same start, 1.70 kg actual | 9.3 | 10.0 | 12.8 |
| Same start, actual floor μ ×0.65 | 17.8† | 11.1 | 13.8 |
| Same start, actual floor μ ×1.40 | 22.0† | 11.1 | 11.1 |
| Selected 5 kg easy 4, 3 kg predictor | 9.8 | 8.3 | 20.9† |
| Fresh easy 1201, 1 kg predictor, 1.7 kg and μ ×1.4 actual | 11.3 | 14.6 | 11.1 |
| Fresh medium 1202, same perturbation | 8.5 | 13.1 | 12.3 |
| Fresh oblique 1203, same perturbation | 25.6† | 51.7† | 36.6† |
| Fresh heavy confirmation 1221, 3 kg predictor, 5 kg actual | 13.8 | 6.4 | 10.6 |
| Fresh heavy confirmation 1222, same perturbation | 33.6† | 9.7 | 6.0 |

The three fresh combined-mismatch scenes were fixed before comparing controllers, with paired optimizer seeds 211/223/227. At nominal 1 kg physics they gave strict counts 2/3, 3/3, 2/3 and mean errors 13.4, 4.8, 13.7 mm. Under combined mismatch each got 2/3, with mean errors 15.2, 26.5, 20.0 mm. At 1203 all three reached or crossed the downrange goal plane; lateral end errors were −23.7/−50.7/−36.6 mm. Knowing the new physics on these **same three execution plants** gave 1/3, 2/3, 2/3 and 9.4/21.3/15.0 mm mean error. Informed ordinary finished 3.0 mm away on 1202 but timed out before settling. Replanning with an accurate model also changes which stochastic trajectory is followed.

At the selected heavy start B settled in 8.9 s versus A's 7.4 s. Peak hand resultants were 49.6/77.6/52.7 N. C missed laterally by 20.7 mm. B's success over C recurred at this geometry; knowing the true 5 kg had previously given B a 2 s success. On the two fresh 5 kg confirmation starts, **B and C both settled both boxes**. Their mean errors were 8.05 and 8.32 mm. A settled one. The extra reachable-offset search accounts for the fresh success-count gain without requiring a learned force input. Two scenes cannot establish equivalence.

## Cone uncertainty

The execution cone has 1.25 times the nominal mass and principal inertias. Geometry, COM, contact parameters, cost, command bounds, eligibility and budgets are fixed. The primary A/Bz/Cz comparison uses 16 × 50 policy calls, three knots, one iteration and the existing ramp and clearance/release cost. The same pose-dependent force shortening constrains the vertical and XYZ matched searches. Results below give strict release time after warmup, or final tilt on failure.

| 4.20 kg actual, 3.36 kg predicted | A | Bz | Cz |
|---|---:|---:|---:|
| Historical away 82, previously favorable CEM 101 | 102.4° | **9.3 s** | 102.5° |
| Same physical scene, historical losing CEM 102 | 102.5° | 102.4° | 102.5° |
| Same scene, fresh CEM 241 | 102.5° | 102.5° | 102.4° |
| Fresh varied 1211, CEM 211 | 102.5°, no valid plan | **5.2 s** | 73.8°, no valid plan |
| Fresh general 1212, CEM 223 | **5.9 s** | **5.5 s** | **6.5 s** |

On 82/101 Bz crossed the loose upright threshold in 8.12 s and finished a 0.54 s strict hold with 41.20 N ground support and zero hand load. Its impulse-weighted hand contact was 6.1 cm up the axis versus 2.2/2.3 cm for A/C. Signed hand angular impulse about the instantaneous ground support was **+11.86 N·m·s** versus −1.34/−2.44. Cone COM translated 48.1 cm. Bz issued 24.6 N mean virtual magnitude, with peak physical hand resultant 76.0 N. Raw workspace projection occurred on 78/72/73% of ticks for A/Bz/Cz. All had over 98% eligible candidate trajectories. Under known 4.20 kg physics only Bz released on this pair as well. Under nominal 3.36 kg, the previous ten-seed challenge gave Bz 4/10 and A/Cz 0/10. The new CEM 241 also failed with nominal physics. Condition-specific reproduction on one geometry does not give independent population evidence.

On 1211 the nominal 3.36 kg results were fail/success/success. With hidden extra mass only Bz released. With the heavier mass disclosed to prediction all three released. Cz under mismatch generated **+13.27 N·m·s** of useful signed impulse, exceeding Bz's +9.48, but lost all valid plans before completing the task. It is possible to rotate the cone and still lose the opportunity for clean release. In the independent XYZ comparison on 1212, both XYZ treatments released with nominal physics. With hidden extra mass XYZ force released and matched XYZ timed out at 81.6°. Ordinary and both vertical treatments released there, so three force axes were unnecessary on that start. Every reported torque/impulse came from replaying executed actuator targets on the saved **execution** physics and verifying its qpos and measured net force. [Wrench method](../tests/cone_contact_wrench.py)

## Prediction error and time

For every selected horizon, restore the execution state including controller history and branch the actual plant on that exact world-space request schedule. Decode and project commands separately at each evolving base pose. Compare the actual-physics branch against the recorded nominal forecast. The first five actual-physics ticks reproduced the executed prefix in every field. The entire identical-physics horizons reproduced qpos, qvel, commands, finger trajectories and forces exactly. The changed-physics numbers include the effect of discarding the solver warm-start. They measure the error of the implemented predictor.

Mean errors across mismatched winning forecasts are given below. Force error is the Euclidean difference of 20 ms mean world-frame gripper-on-object force vectors, including no-contact intervals. Position errors are Euclidean. Cone object XY is the free body origin at the plate. Lookahead 260 ms is the thirteenth policy tick.

| Lookahead | Box XY / finger, mm | Box force error, N | Cone orientation / finger, deg / mm | Cone force error, N |
|---:|---:|---:|---:|---:|
| 20 ms | 0.09 / 0.07 | 0.71 | 0.003 / 0.02 | 0.18 |
| 100 ms | 1.18 / 0.88 | 1.47 | 0.046 / 0.26 | 0.46 |
| 260 ms | 3.62 / 2.78 | 1.96 | 0.201 / 1.08 | 0.70 |
| 500 ms | 6.21 / 4.61 | 2.05 | 0.491 / 2.17 | 0.94 |
| 1 s | — | — | 0.942 / 3.39 | 1.40 |

The all-tick means include much of the approach before contact. At full horizon, among ticks with load in **either** model, box XY error p95 was 27.2 mm and physical force-vector error p95 was 16.3 N. For cone, plate XY error p95 was 31.8 mm and force-vector error p95 was 27.3 N. Contact status disagreed on 8.6% of all terminal box ticks and 1.9% of terminal cone ticks. A contact tick requires at least one 5 ms substep above 1 N. The CSV also gives base translation/rotation, object velocity error, agreement on whether contact occurred within the horizon and onset timing. Replans are correlated, horizon-weighted observations. [Scoring and exact replay](../tests/score_uncertainty.py)

| UniFP loop | Updates | Planning p50 / p95 / max, ms | >100 ms |
|---|---:|---:|---:|
| Box, 51 episodes | 2,886 | 36.8 / 44.1 / 66.5 | 0 |
| Cone, 34 episodes | 2,830 | 49.2 / 59.0 / 91.1 | 0 |
| Sliding-wall pilot, six episodes | 480 | 37.2 / 44.6 / 64.7 | 0 |

The timing includes snapshot transfer, sampling, prediction, scoring, selection, warm start and winning-forecast capture. The full synchronous box cycle including live stepping took 38.7/46.2/71.0 ms p50/p95/max. Model copying and compilation occur at episode setup. Runs used eight available CPUs on a Xeon w3-2435. Physical box/cone peak roll or pitch was 8.56°/9.31°, and all 85 episodes had zero executed nongripper impulse. Box had 14 timeouts; cone had 14 timeouts and three all-invalid stops. Complete per-episode records include failures, settling, contact, projection, validity and latency. [Trial CSV](uncertainty_trials.csv)

## What the robot did at the wall

The historical cost penalizes base XY error and normal finger gap. It leaves finger tangential coordinate and heading free. Thus a fixed world contact point with a turning torso is allowed, as is actual sliding. The recorded motion includes both tangential travel and substantial changes in arm posture. On the positive 45 cm matched-position walk the base moved 43.2 cm, the finger only 14.2 cm, and yaw reached 22.7° from its initial heading. On the negative walk ordinary MPC moved its base −43.5 cm and its finger −51.4 cm with peak heading change 46.8°. An explicitly fixed-point task would require an independently specified world hand anchor.

The optional `--sliding` cost adds squared terminal finger-to-base tangential offset error relative to its warmup value (8 cm unit) and a smooth terminal yaw deviation (15° unit). All body rates and arm targets still compete through CEM. Evaluation additionally asks for at least 85% signed finger travel and peak heading error ≤15°. Six 8 s trials reused the same compliant surface and two starts, seeds, 15 N target, load error objective and budgets as their archived controls.

| Sliding trial | Base / finger travel, cm | Peak yaw change | Duty | Traverse RMSE |
|---|---:|---:|---:|---:|
| Positive, ordinary | 47.2 / 46.0 | 6.0° | 93.4% | 6.12 N |
| Positive, conditioned | 38.3 / 38.0 | 10.4° | 93.6% | 6.24 N |
| Positive, matched | 46.3 / 46.0 | 10.5° | 94.1% | 6.44 N |
| Negative, ordinary | −35.5 / −35.8 | 10.7° | 91.5% | 6.42 N |
| Negative, conditioned | −34.7 / −34.2 | 5.6° | 92.2% | 5.87 N |
| Negative, matched | −38.9 / −40.2 | 11.3° | 90.1% | 6.81 N |

Base/finger tangential displacement disagreement over the trajectory fell from 11.9–27.5 cm RMS to 0.7–1.2 cm RMS. On the negative conditioned run the mean body-frame lateral speed was −0.044 m/s; 31.7 cm of signed finger travel occurred on loaded ticks, and its body-relative lateral endpoint offset changed by less than 1 mm. Finger travel here tracks the physical finger body origin, not material-point slip at the contact patch. All six runs exceeded the existing 4.5 N traverse RMSE threshold. Sliding mechanics improved; force-regulation accuracy remains limiting. [All original and paired measurements](wall_sliding_trials.csv)

## SUMO pilot on paired cone spawns

Both policies received their own neutral 1 s warmup from common global root XY/yaw and identical cone geometry and spawn. Settled cone origin difference was 0.045 mm. Horizontal root displacement between robots was 2.27–2.34 cm and heading differed by 0.94° after warmup. Different base height, joint posture, hand geometry, actuators and low-level policy state are intrinsic to these systems. The pilot uses nominal **3.36 kg** physics for both. Each SUMO trial ran for 12 s after warmup. [Pilot timings and stability](sumo_pilot.csv)

| Start | UniFP ordinary / Z / matched Z first upright | Native SUMO first upright | SUMO at equal scheduled work |
|---|---:|---:|---:|
| Away 82 / CEM 101 | — / 8.04 s / — | 2.04 s | 4.94 s |
| Varied 1211 / CEM 211 | — / 3.78 s / 4.02 s | 1.12 s | 3.66 s |

All four SUMO runs ended within 0.15° of vertical. Native SUMO used 2,400 scheduled policy evaluations per update at 20 Hz, missing its 50 ms period on 465/480 updates. Its two episode p95 values were 69.0 and 68.2 ms. Keeping SUMO's 2 s horizon with eight draws at 10 Hz matched UniFP's 800 scheduled evaluations/update and 8,000/s, with p95 27.7 and 28.9 ms and 0/240 misses. All four runs stayed below 100 ms. These are simulated completion times and measured synchronous planning costs. The returned SUMO state cannot certify substep ground support or gripper-free release. The complete-system upright comparison is meaningful; a strict-release ranking awaits matched physical contact instrumentation. [Direct comparison protocol and box compatibility audit](final_evaluation.md)

## Reproduce

From the repository root, use the pinned UniFP environment. The informed comparisons consume the execution MJB previously generated by the corresponding blind trial. The legacy nominal box parity check also uses the archived force-box records if installed.

```bash
pixi run --locked build
export PYTHONPATH=tools:build OPENBLAS_NUM_THREADS=1
.pixi/envs/default/bin/python tests/check_model_transfer.py
.pixi/envs/default/bin/python tests/model_uncertainty.py box
.pixi/envs/default/bin/python tests/model_uncertainty.py cone
.pixi/envs/default/bin/python tests/model_uncertainty.py box 7 9 11 --informed
.pixi/envs/default/bin/python tests/model_uncertainty.py cone 0 3 --informed
.pixi/envs/default/bin/python tests/score_uncertainty.py
.pixi/envs/default/bin/python tests/wall_sliding.py
pixi run --locked view-wall --padded --force 15 --distance .45 --sliding
```

For the registered ReLIC pilot use SUMO's existing separate environment, with no writes to the reference source or installation. The paired case IDs are 0 for away 82 and 2 for varied 1211.

```bash
for scene in 0 2; do
  PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 ../sumo/.pixi/envs/default/bin/python tests/relic_cone.py --paired-case "$scene"
  PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 ../sumo/.pixi/envs/default/bin/python tests/relic_cone.py --paired-case "$scene" --equal-work
done
```
