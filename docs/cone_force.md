# EE force conditioning for cone uprighting

With the clearance/release objective and 16 candidates, XYZ force released 10 of 12 fresh cones. A vertical position-offset search with zero force also released 10. Full force had one paired win and zero paired losses against its equal-dimensional position-offset control. The advantage was specific to the starting geometry and optimizer seed. Ordinary position remains the cone default.

## What the optimizer selects

CEM still selects world finger XYZ and body vx, vy and yaw rate. In the new nine-column native format the final XYZ is virtual EE force in **the current gravity-aligned body-yaw frame**, in newtons. The legacy eight-column schedule still uses world-horizontal force. The scalar cone treatment uses only signed Z. Full force gets three independent signed draws, each bounded to ±40 N with a requested slew limit of 80 N/s. Each component ramps to the middle knot and holds to the last. The next plan begins at the last executed request. The force RNG is independent of the ordinary hand/base RNG. Histories, actuators, gait and solver state branch from the executed world as before. [Native conversion](../native/world.h) · [CEM sampler](../tools/cem.py) · [Cone integration](../tools/cone_mpc.py)

At every predicted and live pre-action pose, the native controller projects the nominal world request into UniFP's spherical workspace. Its training target here is that projected point plus the yaw-rotated virtual force divided by 200 N/m. There is no scheduled external EE wrench. The native conversion shortens the offset along its ray at the first workspace boundary, checking interior extrema of radius and elevation. Both treatments can reach the same effective endpoint at the same pose. The conditioned actor receives the nominal spherical command and shortened force (observation scale 0.01). The matched actor receives the displaced spherical command and zero force. On diverging trajectories the projection is recomputed independently at each actual base pose. The moving boundary can shorten the issued force faster than the requested slew. The comparison excludes extra reward-target reach outside the ordinary workspace. [Training reward](../../unifp-spot/unifp_spot/envs/base/mdp/rewards.py) · [Training limits](../../unifp-spot/unifp_spot/envs/spot/spot_pos_force_cfg.py)

All five controllers optimize the same running cone alignment and proximity cost, release guidance and 4 cm clearance penalty. Candidates with native failure or nongripper cone impulse are ineligible. One CEM pass scores 16 candidates × 50 policy ticks, fits three elites, and executes the winning five-tick prefix. Three linear knots, existing noise ramp, eight workers and a 12 s episode limit apply throughout the main comparison. Strict success requires 25 consecutive 20 ms ticks within 10°, angular speed ≤0.25 rad/s, at least half the cone weight in ground support, zero active gripper contact, sufficient base height and stable roll/pitch. Every episode checks executed poses, velocities, actions, issued commands, hand and ground forces and gripper touch against the selected native prefix.

## Restored contact interventions

Six executed states from near 0 and general 82 were restored with the archived nominal arm/base requests. Each received 0.5 s of zero force or ±12 N on one yaw-frame axis, and each signed force had a matched ordinary-position twin. These deliberate steps measure policy response. The planner uses ramped requests. The table gives **degrees of tilt removed** in that half-second. Each pair is conditioned / matched. Near 0's shaft-contact window includes both lower shaft and plate contacts.

| Window | Ordinary | −X | +X | −Y | +Y | −Z | +Z |
|---|---:|---:|---:|---:|---:|---:|---:|
| Approach | 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| Near, first loaded shaft/plate | 21.3 | 21.3 / 21.3 | 31.6 / 34.1 | 16.2 / 25.6 | 21.3 / 21.3 | 21.3 / 21.3 | 27.6 / 20.3 |
| Near, partially upright | 31.0 | 20.7 / 15.2 | 31.0 / 31.0 | 31.0 / 31.0 | 21.0 / 25.7 | 31.0 / 31.0 | 12.8 / 22.8 |
| Near, unloaded at 22.4° | 17.4 | 17.4 / 17.4 | 17.4 / 17.4 | 17.4 / 17.4 | 17.4 / 17.4 | 17.4 / 17.4 | 17.4 / 17.4 |
| Away, plate only | ≈0 | ≈0 / ≈0 | ≈0 / ≈0 | ≈0 / ≈0 | ≈0 / ≈0 | ≈0 / ≈0 | ≈0 / ≈0 |
| Away, productive shaft/plate | 17.9 | 17.9 / 18.1 | 17.9 / 17.9 | 17.9 / 17.9 | 11.9 / 15.7 | 17.9 / 17.9 | 15.0 / 18.8 |

Positive Z offered a measurable conditioning-specific gain in the first loaded window and lost it later. This motivated freezing Z as the scalar search before the held-out experiments. Positive X also accelerated initial tipping, but the reachable ordinary offset accelerated it further. The tested lateral perturbations offered little evidence of an advantage. Several directions were effectively blocked by the workspace. The six windows cover only two scenes and give no general axis ranking.

The hand impulse was reconstructed from individual contact wrenches at 200 Hz, with archived action replay checked against every native pose and net hand force. For the support moment we used the normal-load-weighted center of the cone's active ground contacts and projected hand torque on the unit vector along cone-axis × world-up. It measures the hand contribution about a moving support region and excludes gravity and ground torque. In the stalled plate window ordinary / +Z / matched produced 0.93 / 3.21 / 0.72 N·s of hand impulse just 2.3–2.4 cm up the cone axis. Their respective signed support angular impulses were −0.40 / −0.90 / −0.17 N·m·s. All three left tilt essentially unchanged. On the near partial cone the same treatments delivered +1.29 / +0.59 / +1.03 N·m·s with 31.0 / 12.8 / 22.8° of tilt improvement. [Contact probe](../tests/cone_force_snapshots.py) · [Pair-wrench integration](../tests/cone_contact_wrench.py)

## Held-out closed-loop results

Scene seeds 301–304 were near, 305–308 varied and 309–312 general. Each tier used independent optimizer seeds 71, 79, 83 and 89 in order. General seeds 309 and 310 put the cone about 66° left and 69° right of Spot, respectively. Seed 312 pointed the tip just 7.4° from the outward ray. The five runs for each condition shared the scene, optimizer seed, warmup and computational budget. All failures contribute to mean final tilt and translation. The time columns average over each controller's strict successes, so their subsets differ. In this set the loose upright and strict released counts coincided.

| Treatment | Released | Final tilt, mean | First upright / strict release (s) | Final COM displacement XY (m) | Eligible candidates | Plan p50 / p95 / max (ms) | >100 ms |
|---|---:|---:|---:|---:|---:|---|---:|
| Ordinary position | 8/12 | 29.9° | 5.52 / 7.06 | 0.37 | 81.1% | 48.9 / 61.9 / 87.4 | 0/843 |
| Z force | 9/12 | 25.6° | 5.48 / 6.99 | 0.40 | 85.4% | 48.0 / 59.9 / 75.3 | 0/989 |
| Matched Z offset | 10/12 | 14.9° | 5.50 / 6.94 | 0.42 | 86.6% | 50.0 / 60.3 / 72.4 | 0/862 |
| XYZ force | 10/12 | 17.0° | 4.92 / 6.40 | 0.43 | 82.5% | 49.4 / 60.1 / 70.2 | 0/880 |
| Matched XYZ offset | 9/12 | 20.0° | 4.06 / 5.44 | 0.41 | 85.8% | 48.9 / 60.1 / 113.3 | 1/693 |

The full conditioned search gained seeds 303 and 309 over conditioned Z, and lost 302. Versus matched XYZ, the sole additional success was 309. Z conditioning gained 306 over matched Z and lost 303 and 309. Every controller timed out on away-pointing 312. Matched Z and full force reached the same success count by solving different cases. At this sample size, a single case changes the rate by 8.3 percentage points.

Across the held-out episodes the mean of the episode peak physical hand loads (norm of each 20 ms net force) was 67.1 / 83.0 / 74.0 / 88.0 / 67.3 N in table order. Mean integrated resultant impulses were 86.2 / 73.4 / 84.0 / 86.4 / 56.2 N·s. These tick-resultant impulse magnitudes can be smaller than a sum over individual pair wrenches. Z force requested 24.5 N mean magnitude and issued 8.8 N to the actor. XYZ requested a 35.7 N mean vector norm and issued 19.9 N. Projection shortened the requests by over 0.01 N on 69% and 61% of ticks, respectively, averaging episode fractions. Peak robot roll/pitch across the set was 9.54°. The virtual inputs are distinct from those measured physical loads.


## Historical controls and seed sensitivity

On the original seven starts, ordinary position released 6/7, signed Z force 7/7, matched Z 5/7, XYZ force 7/7 and matched XYZ 6/7. The ordinary run reproduced all common non-timing arrays in all seven archived clearance-controller episodes exactly. With a second optimizer seed (67), all five controllers released near cone 1. On general 82 the cone plate was 1.60 m away at +53.2° relative bearing and its tip pointed outward. Changing just the CEM seed changed the outcome.

| General 82, 16 candidates | Ordinary | Z force | Matched Z | XYZ force | Matched XYZ |
|---|---:|---:|---:|---:|---:|
| CEM seed 43 | timeout | timeout | timeout | timeout | timeout |
| CEM seed 67 | timeout | released 9.1 s | timeout | released 9.7 s | timeout |

At seed 67 the Z and XYZ controllers first crossed the loose upright threshold at 7.86 and 8.08 s. They then held within strict tolerances for 0.54 and 0.52 s with zero final hand load, 32.96 N vertical ground support and negligible final angular speed. The cone COM ended 0.47 m from its starting XY position in both episodes. Robot peak roll/pitch was 8.52° and 7.84°. The physically measured peak tick-mean gripper forces reached 75.2 and 78.6 N. Neither forced an externally specified lift or contact point. At seed 43 full force produced a 50.9 N peak while the cone remained fallen. [Released XYZ trajectory](../artifacts/cone_force/general-82/67-16-conditioned-3/episode.npz) · [Stalled XYZ trajectory](../artifacts/cone_force/general-82/43-16-conditioned-3/episode.npz)

Across held-out full-force runs, issued X/Y/Z component RMS values were 15.8/12.8/14.8 N. All three changed sign and exceeded 1 N in magnitude simultaneously on 54% of the 4,400 executed ticks. In 512 independent mixed-XYZ requests at a rotated base pose, 284 required shortening. Native conditioned and matched targets agreed within 0.089 µm in world space; intermediate segment samples and reversed candidate order also passed. Across the 120 trials all 20 distinct scenes reproduced identical warmup poses, actions and policy commands across treatments, and executed nongripper impulse was zero. The 50 Hz comparison uses the pose before issuing the command. [Projection check](../tests/check_cone_force.py)


The complete successful XYZ episode produced 35.21 N·s of total gripper impulse over plate, foot and lower-shaft contacts with an impulse-weighted axial location of 5.0 cm. The failed episode produced 8.21 N·s entirely at plate/foot contacts 2.25 cm up the axis. Signed hand angular impulse about the ground support was +6.42 versus +0.067 N·m·s. Near the end of the archived successful tipping trajectory, both force and offset interventions could still release under a common continuation. On the near initial shaft window, replaying the remaining original position-only schedule after the +Z intervention failed to release for both treatments, despite faster short-term tipping in the conditioned branch. These continuations hold subsequent commands fixed and can incur collisions once the trajectory diverges. The MPC episode results above use independent replanning. [All 78 signed and ordinary snapshot branches, including continuation results](cone_force_snapshots.csv) · [Full episode wrench decomposition](../artifacts/cone_force/seed82_wrenches.json)

### Exploration and planning budget

Once per simulated second we restored the exact executed state on the two 82 XYZ-force runs and evaluated the same 16 hand/base proposals with XYZ force, equivalent ordinary offsets, Z force only, or zero force. On the stalled run, 74 of 184 eligible conditioned candidates predicted gripper contact and zero predicted 10° of improvement. The best predicted gain was 0.21°. Removing force reduced predicted contacts to 23/188 without uncovering useful rotation. On the successful run, all 16 same hand/base candidates at 7.0 s forecast more than 10° progress with XYZ, matched XYZ **and zero force**. Both the full-force and zero-force best gains were 71.8°. The role of earlier decisions in reaching productive geometry is substantial. The probe cannot ascribe the subsequent tip to a unique indispensable force command. [Restored population comparisons](cone_force_populations.csv) · [Diagnostic](../tests/cone_force_population.py)

As all five 16-candidate controllers failed on 82/43, we compared all five at 24 candidates on that case and on historical near 1/17. All five still timed out on 82/43, and all five released near 1/17. At 16 candidates all four augmented searches already solved near 1 and ordinary position timed out. Increasing the budget alone did not rescue the away-side failure. Across 110 main episodes, 16 candidates (800 policy evaluations/update) took 49.5 / 60.3 / 113.3 ms p50 / p95 / maximum with 1/8,110 deadline misses. The ten 24-candidate episodes (1,200 evaluations/update) took 72.1 / 84.7 / 106.2 ms with 1/875 misses. These are complete headless replans pooled over updates, including all-invalid batches, on eight workers. Different task mixtures limit direct timing comparisons. The held-out comparison gives no consistent indication that three dimensions overwhelm the search at 16 candidates; geometry, workspace clipping and seed sensitivity remain limiting. [Per-episode performance, physical measurements and latency for all 120 runs](cone_force_trials.csv)

## Towing calibration

We repeated the 12 previous physical tether conditions, adding one prespecified +0.016 m/s ordinary correction along each walking direction to the ordinary matched velocity. All three were scored against the **original common effective target** over the same 3 s interval. No value was fitted to the loaded outcomes. The baseline and conditioned rows use their archived trajectories with identical warmup poses checked on every new calibrated run.

| Treatment | Loaded velocity RMS error (m/s) | Mean along-track speed (m/s) | Box progress at 3 s (m) | Reached 25 cm |
|---|---:|---:|---:|---:|
| −16 N virtual base force | 0.04251 | 0.19630 | 0.45984 | 12/12 |
| Matched ordinary velocity | 0.04861 | 0.18169 | 0.41503 | 11/12 |
| Ordinary velocity +0.016 m/s correction | 0.04279 | 0.19629 | 0.45965 | 12/12 |

Calibration reduced tracking error in all 12 comparisons with uncorrected ordinary velocity. Against the conditioned controller it won four paired cases. The residual mean RMS-error gap was 0.00028 m/s. Mean time to 25 cm was 2.035 versus 2.033 s for calibrated and conditioned. The previously observed mean towing benefit is largely reproducible by an ordinary command adjustment. [Paired towing measurements](tow_velocity_correction.csv) · [Calibration probe](../tests/tow_velocity_correction.py)

## Next experiment and reproduction

Retain the 16-candidate ordinary default and keep Z/XYZ as experimental modes. The most informative next comparison is to replan repeatedly from the same **pre-contact** seed-82 snapshots with fresh paired optimizer seeds and measure how often each mode reaches a productive support moment and a strict release. This separates discovering a better approach from applying force after useful contact already exists. Future box comparisons should freeze the schedule and world-space goal on development cases, then score settled XY error, overshoot, undershoot, lateral drift and time to settle on held-out cases. The historical box displacement results characterize fixed commands only.

With the prepared assets, exported policy and archived snapshot episodes, run these commands from the repository root.

```sh
pixi run --locked build
export PYTHONPATH=build:tools OPENBLAS_NUM_THREADS=1
.pixi/envs/default/bin/python tests/check_cone_force.py
.pixi/envs/default/bin/python tests/cone_force_snapshots.py
for split in difficult held original; do
    .pixi/envs/default/bin/python tests/cone_force_trials.py "$split" z
done
.pixi/envs/default/bin/python tests/cone_force_trials.py budget z --count 24
.pixi/envs/default/bin/python tests/cone_force_population.py
.pixi/envs/default/bin/python tests/tow_velocity_correction.py
```

The three historical easy-4 box treatments reproduced their archived actions, hand loads, positions and velocities exactly. The existing cone branching/contact, wall-force, base-load and tether checks passed. [Success video, 82/67](../artifacts/cone_force/general-82/67-16-conditioned-3/episode.mp4) · [Failure video, 82/43](../artifacts/cone_force/general-82/43-16-conditioned-3/episode.mp4)
