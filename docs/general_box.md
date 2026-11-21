# Full-command MPC on longer approaches and sideways pushes

Use **full CEM, 24 candidates, one iteration and a 0.5 s horizon** as the randomized viewer and experiment baseline. It precision-settled five of seven distinct starts, including a box 2.43 m away and a 66 cm push starting 74° off heading. A second elite update improved development errors, but lost one precision success on the three reserved starts. All seven default runs together had 36.2 / 43.4 / 57.8 ms median / p95 / maximum planning latency and zero misses in 1,019 updates.

## Tasks and scoring

| New tier | Robot to box center | Absolute box bearing from robot heading | Absolute push angle from approach | Goal displacement | Control limit |
|---|---:|---:|---:|---:|---:|
| Extended | 1.75–2.45 m | 0–25° | 0–20° | 25–55 cm | 15 s |
| Crossing | 1.35–1.90 m | 30–70° | 25–65° | 25–55 cm | 15 s |
| General | 1.45–2.30 m | 55–110° | 45–110° | 30–70 cm | 20 s |

The approach vector points from robot to box. Radius, angle magnitudes and displacement are uniform within each range. The angles have independent random signs, so the robot can need to turn in one direction to approach and another to push. World heading spans 360°, robot XY spans ±0.4 m per axis, and box yaw spans ±90° relative to initial robot heading. Floor–box and hand–box sliding coefficients span 0.18–0.48 and 0.20–0.54 independently, with matched live and prediction physics. The generator checks spawn contacts and the runner checks the one-second warmup. Goal offsets are applied to the executed box center after warmup. Original tiers retain their seeded poses and nine-second limit. [Scenario generator](../tools/box_scenario.py)

Settling requires 25 consecutive 50 Hz box XY samples inside the radius and speed ≤2 cm/s throughout the last five samples, assessed at each executed prefix end. The runner records the first qualifying control time at **50, 15 and 10 mm**. Coarse settling also requires closing at least half the starting error throughout the hold. Precision settling still terminates the episode, so 10 mm counts only when achieved by that cutoff. A blank time in the data means the criterion was never observed. Progress is signed box displacement projected onto the requested axis, divided by requested length. It can exceed 100% or conceal lateral drift, so read it alongside final XY error. [Task loop](../tools/box_mpc.py) · [Per-episode measurements](general_box_trials.csv)

## Matched experiments

Development used extended 10, crossing 11 and general 12, all with CEM seed 17. All three precision-settled with the reference, so we added general 9 based on its geometry to stress a push turning back across the robot (94° approach bearing and another 105° to the goal direction). The screen paired the long approach and this difficult case. The two finalists were then compared across all four development starts. Reserved tasks were extended 60, crossing 61 and general 62, with independent CEM seeds 43, 47 and 53. The primary held-out comparison was 24×1 versus 16×2, both at 0.5 s. We also ran the unchanged 1 s configuration on the same three cases in a secondary comparison because of its development coarse-settling result. There was no subsequent tuning.

Each planner uses the same bounded directional objective, unless indicated, three linearly interpolated knots and a 0.1 s execution prefix. CEM samples EE world XYZ and the three body rates together. Each rate reaches its sampled value at half the horizon and holds it. Longer horizons therefore also change the time available to ramp each command. EE slew and body-rate slew use the physical knot spacing. Warm starts shift by five executed policy ticks for every horizon. All comparisons used eight native workers on a Xeon w3-2435. Latencies pool individual headless planning updates across the listed cases and include snapshot, sampling, all rollouts, scoring and fitting. The workload column gives the scheduled policy evaluations per update (candidates × iterations × horizon × 50). Four physics steps follow each policy call. [Sampler](../tools/cem.py) · [Native rollouts](../native/rollouts.h)

| Screen on extended 10 and general 9 | Horizon | Policy calls/update | Settled 50 / 15 / 10 mm (of 2) | Final error per case, mm | Plan median / p95 / max, ms |
|---|---:|---:|---:|---:|---:|
| 24×1 | 0.5 s | 600 | 1 / 1 / 1 | 1.9 / 87.5 | 36.3 / 42.8 / 48.1 |
| 16×1 | 0.5 s | 400 | 1 / 1 / 1 | 3.7 / 117.3 | 24.3 / 29.5 / 33.6 |
| 8×3 | 0.5 s | 600 | 0 / 0 / 0 | 44.3 / 216.6 | 39.0 / 45.5 / 52.4 |
| 8×1 | 1.5 s | 600 | 0 / 0 / 0 | 300.5 / 272.1 | 35.0 / 43.7 / 54.9 |
| 32×1 | 0.5 s | 800 | 0 / 0 / 0 | 79.5 / 166.5 | 47.1 / 55.9 / 63.8 |
| 16×2 | 0.5 s | 800 | 1 / 1 / 1 | 0.8 / 38.4 | 49.4 / 58.0 / 67.5 |
| 16×1 | 1.0 s | 800 | 1 / 0 / 0 | 104.9 / 24.8 | 46.3 / 53.5 / 58.5 |
| 24×1 | 1.0 s | 1,200 | 2 / 1 / 0 | 6.3 / 14.6 | 67.6 / 78.3 / 87.8 |
| 24×1, center proximity | 0.5 s | 600 | 1 / 0 / 0 | 40.0 / 313.2 | 36.1 / 43.6 / 56.5 |
| 24×1, three elites + noise ramp | 0.5 s | 600 | 1 / 1 / 0 | 7.5 / 117.2 | 38.1 / 60.1 / 71.0 |

SUMO's Spot overrides use 24 rollouts, three nodes, three elites, a 3.5 noise ramp and 2 s prediction. Its simplified pushing reward averages object–goal and gripper–center distances and penalizes squared mean object speed. Judo's reset samples a 1–2 m annulus with additional world XY noise and independent robot translation. Here the polar bounds give directly reproducible relative placement. We compared the existing center-proximity ablation to directional guidance with identical local-goal, spacing, validity and CEM settings. Both average running and terminal squared box error and advance the local target up to 8 cm along the push line. We kept the precision objective fixed when measuring coarse success. [SUMO reward](../../sumo/sumo/tasks/spot/spot_push.py) · [SUMO optimization](../../sumo/sumo/controller/optimizer_overrides.py) · [Judo reset](../../judo/judo/tasks/spot/spot_box_push.py) · [Judo CEM](../../judo/judo/optimizers/cem.py)

| Split | Candidates × iterations × horizon | Settled 50 / 15 / 10 mm | Mean final error including timeouts | Plan median / p95 / max | Misses >100 ms |
|---|---|---:|---:|---:|---:|
| Development, 4 starts | 24×1 × 0.5 s | 3 / 3 / 2 | 26.1 mm | 36.3 / 42.8 / 48.1 ms | 0/580 |
| | 16×2 × 0.5 s | 3 / 3 / 2 | 12.8 mm | 49.6 / 58.5 / 69.3 ms | 0/496 |
| | 24×1 × 1.0 s | 4 / 2 / 1 | 17.9 mm | 68.1 / 79.9 / 89.7 ms | 0/638 |
| Held out, 3 starts | 24×1 × 0.5 s | 2 / 2 / 1 | 78.9 mm | 36.0 / 44.4 / 57.8 ms | 0/439 |
| | 16×2 × 0.5 s | 2 / 1 / 0 | 120.3 mm | 48.8 / 58.3 / 70.5 ms | 0/434 |
| | 24×1 × 1.0 s | 2 / 1 / 0 | 193.8 mm | 66.6 / 79.5 / 84.5 ms | 0/438 |

The reference precision-settled held-out extended 60 at 6.8 mm in 14.8 s and crossing 61 at 14.95 mm in 9.1 s. On general 62 (1.89 m approach at 93°, 63 cm push another 102° across), it timed out at 214.8 mm after 20 s. It achieved 67.7% downrange progress, with 112 mm peak lateral deviation. Two iterations reached contact 3.32 s earlier but finished at 310.4 mm. The longer horizon finished at 544.0 mm. Across the 35 episodes there were zero planning deadline misses in 5,342 updates, zero executed nongripper impulse, peak absolute roll/pitch of 8.97°, and 239 ineligible candidates in 127,608 sampled trajectories. Every update had a valid plan. All failures were timeouts. These seven starts provide a comparison on matched physics and tasks, with limited coverage of each tier.

## Contact and optimization limits

- Before the first update on both general 9 and 62, all 24 candidates predicted zero contact and identical box XY. Directional cost still spanned 0.0000108 and 0.0000126 m² from predicted approach and body spacing. Both robots reached contact. On general 9 the 0.5 s reference contacted at 8.34 s, the 1 s reference at 7.70 s, and the equal-work eight-candidate 1.5 s planner at 13.54 s. Extra lookahead with fewer samples did little to discover useful initial motion in that case.
- Center proximity acquired contact later on both screened starts. On general 9 it yielded nine contact ticks against 58 with directional guidance. At 7.7 s in an exact replay of the directional controller, 20/24 candidates predicted contact. Directional and center costs selected the same candidate with 13.5 mm predicted final-goal gain, while another eligible candidate predicted 52.0 mm. At 19.5 s even the best eligible candidate by immediate final-goal gain predicted a 0.48 mm loss. Both the choice of local target and the available short-horizon motions can limit progress. Counterfactual gains here cover a single forecast.
- On held-out general 62, the reference first contacted at 8.88 s, turned 145° over the episode, and had gripper contact in 127 policy ticks, including the final 0.02 s before timeout. Its projected EE radius was at the 0.55 m limit on 76% of control ticks. At 18 s, all 24 candidates predicted gripper contact, and their best eligible final-goal gain was 39.2 mm against 13.8 mm for the directional winner. Reaching the box and finding contact were possible. The experiment leaves open how much of the residual error comes from working pose, short-horizon sampling and the 8 cm local target. A useful next comparison is to vary only that local target on the same executed contact snapshots and measure subsequent closed-loop settling. [Contact-rank replay](../tests/inspect_recovery.py) · [Held-out timeout video](../artifacts/general_box/baseline/general62/episode.mp4)

## Reproduction and checks

From the repository root with the runtime artifacts prepared:

```bash
pixi run --locked view-box --difficulty general --scenario-seed 12 --cem-seed 17 --episodes 1
pixi run --locked python tools/box_scenario.py --difficulty extended --seed 60 --out artifacts/general_box/scenes/extended60
pixi run --locked env OPENBLAS_NUM_THREADS=1 PYTHONPATH=build python tools/box_mpc.py --scenario artifacts/general_box/scenes/extended60 --seed 43 --out artifacts/general_box/replay60
```

Use `--candidates 16 --iterations 2` for the two-pass comparison, `--horizon 1` for the longer reference, `--objective proximity` for the center cost, or `--sumo-sampling` for the combined ramp/elite ablation. The viewer also accepts these options and an explicit `--seconds`. Continuous playback increments scenario seeds and resets the optimizer independently.

The actual paced window completed general 12 with the same 9.05 mm error, 175 contact ticks, settling times and 149 plans as headless. Median / p95 / max planning was 51.7 / 61.5 / 72.8 ms with zero misses, 15.9 simulated seconds including warmup in 16.31 wall seconds and 29.4 presented poses/s. All matched experiments had identical warmup states and goals, zero force commands and exact selected-prefix reproduction. Viewer paths were checked on all three new tiers at 0.5, 1 and 1.5 s. The archived 8, 35 and 50 cm demos reproduced every saved non-timing trajectory and plan array. [Executed side approach and 66 cm push](../artifacts/general_box/baseline/general12/episode.mp4) · [Viewer path and reset checks](../tests/check_view_box.py)
