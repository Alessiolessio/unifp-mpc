# Guided CEM for UniFP base and arm commands

`--base guided` optimizes world EE XYZ and body vx, vy and yaw rate with the pretrained policy in every forecast. It shares the bounded box objective, 0.5 s horizon, three knots, 0.1 s prefix and eight native workers with the forward, stance and full baselines.

At 24 candidates and one CEM iteration, each update uses the following allocation.

1. Evaluate 16 joint trajectories from one executed-state snapshot. Keep the shifted incumbent and elite mean. Give one sample the exact stance rates and center alternate lateral/yaw perturbations around the geometric or fitted rates. CEM samples EE and forward speed throughout this batch.
2. Keep the best valid trajectory's EE and vx fixed for eight comparisons from the same snapshot. Test exact stance, each stance axis combined with the other winning axis, and five joint perturbations around the winning and stance rates. If the first batch has no valid candidate, spend the second batch on fresh joint draws.
3. Fit two elites across the combined population. Keep the best evaluated valid plan across both batches and all requested iterations. Execute its first five policy ticks and shift the physical-command warm start. With fewer than eight candidates, use a single guided joint batch.

The two auxiliary rates can vary independently. Each body axis still has one target rate per candidate, joined linearly to the last executed request and held after the middle knot. Bounds remain ±0.35/0.22 m/s and ±0.50 rad/s, with 0.8 m/s² or rad/s² slew per axis. Guidance enters through candidate generation and is recomputed at each replan. The native pre-action world-to-body projection runs on every candidate's predicted pose. [Sampler](../tools/cem.py) · [Planner](../tools/box_mpc.py)

## Closed-loop comparison

Final box-goal error in mm. Bold entries met the original 15 mm, 0.5 s hold and final-speed criteria within 9 s of control. Warmup states and goals matched exactly across each row. Oblique 2/3 were reserved from development here and had appeared in the earlier stance study. Oblique 4/5 and optimizer seed 41 were fresh for this study.

| Scene / optimizer seed | Forward | Fixed stance | Full CEM | Guided CEM |
|---|---:|---:|---:|---:|
| Hard 9 / 17 | 19.3 | **9.9** | 57.4 | 70.6 |
| Long 14 / 29 | 50.7 | 100.8 | 95.3 | 107.2 |
| Oblique 0 / 17 | 408.3 | **3.1** | 19.0 | **2.8** |
| Oblique 1 / 17 | 245.0 | 83.1¹ | **6.3** | **3.6** |
| Held-out oblique 2 / 29 | 358.3 | 26.7 | **11.4** | **2.6** |
| Held-out oblique 3 / 29 | 260.5 | 22.6 | 48.7 | 107.9 |
| Held-out oblique 4 / 41 | 337.9 | **0.2** | **6.2** | **5.0** |
| Held-out oblique 5 / 41 | 419.9 | 6.3² | **14.2** | 17.3 |
| Settled, all / held-out | 0/8 · 0/4 | 3/8 · 1/4 | 4/8 · 3/4 | 4/8 · 2/4 |

¹ No valid plan at 3.9 s. ² Only 0.42 s inside tolerance at timeout.

Guided CEM made gripper contact on all eight starts. On oblique 3 it continued making contact up to 8.92 s, ending 77 mm past the target along the push and 76 mm off line. Those contacts left both downrange and lateral error.

| Pooled over eight episodes | Valid / evaluated | Peak roll or pitch | Planning p50 / p95 / max, ms | 100 ms misses |
|---|---:|---:|---:|---:|
| Forward | 17,280 / 17,280 | 8.4° | 35.0 / 41.7 / 52.4 | 0/720 |
| Fixed stance | 13,492 / 13,680 | 9.5° | 36.2 / 42.9 / 53.4 | 0/570 |
| Full CEM | 14,112 / 14,112 | 8.5° | 36.1 / 43.0 / 51.6 | 0/588 |
| Guided CEM | 14,949 / 14,952 | 9.0° | 37.7 / 44.8 / 52.9 | 0/623 |

Validity includes native trajectory validity and the mobile cost's nongripper collision rejection. Executed nongripper impulse was zero in every episode. Timing covers the snapshot, both sampling and evaluation passes where applicable, costs, elite fit, warm start and prefix selection. All values are headless. Selected prefixes exactly reproduced forecast policy commands, actions, poses and gripper forces.

## Authority and contact recovery

We replayed every winning guided trajectory from its executed snapshot, then replaced only vy/yaw with exact stance guidance anchored to that state's last issued rates. The selected lateral target differed on 373/623 updates, and yaw differed on 445/623. Mean absolute target differences were 0.032 m/s and 0.065 rad/s. The selected trajectory had lower predicted cost on 522 updates and tied to within 1e−10 on 101. The explicit exact-stance candidate makes this a check of selection and use of the extra authority. Independent closed-loop replanning produced the task outcomes above. [Replay check](../tests/check_guided_base.py)

| Archived slip snapshot | Forward | Fixed stance | Full CEM | Guided CEM |
|---|---:|---:|---:|---:|
| Hard 9, 4.0 s | 19.3 | 68.6 | 39.5 | 42.6 |
| Hard 9, 5.5 s | 19.3 | 87.1 | 54.9 | 33.6 |
| Long 14, 4.0 s | 50.7 | **4.6** | 127.2 | 132.6 |
| Long 14, 6.5 s | 50.7 | 13.7 | 47.8 | 22.4 |

Each branch restored the physical state, policy history, gait, delays and shared optimizer coordinates from the archived forward run. Each had the remaining time until the original 9 s deadline. On the late long-14 cut, guided CEM re-established corrective contact in 0.46 s versus 0.76 s for full CEM and 0.94 s for forward. Fixed stance also took 0.46 s. All four guided branches timed out. A corrective tick requires measured gripper contact, impulse toward the final goal and a simultaneous fall in box-goal distance. [Recovery runner](../tests/recover_box.py)

## Budget choice

Development used hard 9, long 14 and oblique 0/1. A one-pass mix of warm and stance-centered samples settled one of four. Pairing half the batch with identical EE/vx isolated base choices and settled one. Carrying fitted base residuals to the next geometric nominal and reserving per-axis alternatives on the previous winner each settled one. Two 12-candidate paired CEM iterations settled both development obliques, with episode p95s of 46.3–48.1 ms. Refining the current joint winner with 16 + 8 settled the same two with lower final error on all four development cases and used three eight-worker waves. We froze that sampler before the held-out runs. The number of evaluations, horizon, command knots and physics fidelity stayed fixed.

Guidance is useful for some oblique approaches. Full CEM remains the stronger comparator on the held-out success count and fixed stance retains a recovery the sampled controllers lost. The next diagnostic is to inspect why the best short-horizon correction on oblique 3 or long 14 accumulates cross-track error in execution.

## Reproduction

From the repository root with the existing assets and native build, run

```sh
pixi run --locked view-box --base guided --difficulty oblique --scenario-seed 1 --cem-seed 17 --episodes 1

export PYTHONPATH=build:tools OPENBLAS_NUM_THREADS=1
pixi run --locked python -B tests/run_recovery_scene.py oblique4 oblique5 --seed 41 --modes forward stance full guided --out artifacts/guided/replay
pixi run --locked python -B tests/check_guided_base.py artifacts/guided/replay/*_guided_bounded_41
pixi run --locked python -B tests/recover_box.py --modes forward stance full guided
```
