# Cone exploration at a fixed rollout budget

**Retain ordinary position/velocity MPC as the default.** Four correlated hand/base proposals are worth keeping as an option. On six fresh general cones ordinary MPC strictly released three with local sampling, four with a uniform mixture and six with correlated proposals. A random-stream control released five. On the principal away-pointing cone, vertical conditioning showed a repeatable, scene-specific benefit. Full XYZ conditioning did not reproduce its historical advantage across ten fresh optimizer seeds.

## Paired protocol

Each scene settled for 50 neutral policy ticks. We restored that contact-free native snapshot before each treatment and started the optimizer with the warmup world target, zero base/force requests and the same initial knot statistics. This tests **discovering contact starting before the approach**. The snapshot includes integration and constraint state, full policy history, previous action, actuator delays and gait timing. Separate invocations rebuilt the same scene; we compared the complete warmup pose, velocity, action, policy command and hand-force arrays across all 165 episodes and nine scenes. Each was identical within its scene. Force modes use a separate sampling stream for the force degrees of freedom.

The clearance/release objective, hard nongripper collision rejection, three linear knots, 3.5 SUMO noise ramp, three elites, 16 candidates, one iteration, eight native workers, 1 s forecast and 100 ms execution prefix were fixed. Each update evaluates 800 policy steps. The box, cone and force defaults retain their earlier settings. [Shared planner](../tools/box_mpc.py) · [Sampler](../tools/cem.py) · [Cone objective and proposals](../tools/cone_mpc.py)

- **Local** uses the incumbent, elite mean and the existing Gaussian draws.
- **Broad** replaces the last four lanes with independent uniform hand/base endpoints over the normalized workspace, held across the two future knots before bounding. A separate random stream preserves the first 12 local draws at a common optimizer state. Force components, when enabled, retain their original draws. All candidates use the same start knot, slew limits, projection and cost.
- **Joint** uses the existing four correlated proposals. Each samples a hand location along the cone with Gaussian spread and a compatible base translation/turn. Every command competes in the native evaluator.
- **Sham** draws and discards the joint proposals, leaving all 16 Gaussian/incumbent/mean lanes. This matters because the existing joint sampler consumes the main RNG before drawing local candidates. Local versus joint changes both proposal content and the realization of the other samples. The sham comparison isolates replacement of the four lanes within that random stream.

The principal diagnostic was general cone 82, CEM seeds 101–110. The other diagnostic starts were general 411/117 (60° plate bearing, sideways tip) and 412/117 (near-zero bearing, outward-oblique tip). The first fresh validation block used 418/131, 419/137 and 421/139. A replication block used 443/151, 444/157 and 445/163. These six general cones spanned plate distances 1.24–1.59 m and relative bearings −72° to +73°. We kept all outcomes. Each fresh scene has one paired optimizer seed; ten repeats of cone 82 characterize optimizer variability conditional on that scene.

## What happened on cone 82

Productive contact here means that the executed cone gained at least 10° toward upright after first gripper contact. Time starts after warmup. Strict success additionally requires the existing ground-supported, low-speed, hand-free hold within 10° for at least 0.5 s with stable Spot. Axial location is the hand-impulse-weighted projection from the cone base along the cone axis. The signed hand angular impulse uses the instantaneous normal-load-weighted ground support point and the direction increasing axis alignment. It is a hand contribution, not the full change in object angular momentum. [200 Hz wrench reconstruction](../tests/cone_contact_wrench.py)

| Optimizer, local sampler | Productive / strict releases | Successful CEM seeds | Mean final tilt, all ten | Mean valid candidates | Ticks with raw position projected >10 µm |
|---|---:|---|---:|---:|---:|
| Ordinary position | 0 / 0 | — | 102.4° | 99.0% | 85.7% |
| Vertical force | 4 / 4 | 101, 104, 105, 108 | 61.5° | 98.3% | 74.1% |
| Matched vertical position | 0 / 0 | — | 102.4° | 98.9% | 76.7% |
| XYZ force | 0 / 0 | — | 102.4° | 98.9% | 75.7% |
| Matched XYZ position | 2 / 2 | 101, 107 | 82.0° | 98.7% | 68.1% |

Vertical force gained four paired releases and lost none against matched vertical. The two-sided exact sign test on the four discordant pairs gives 0.125. Full XYZ lost two and gained none against matched XYZ. Historical seeds 43 and 67 were excluded from these counts. On lateral scene 411, the local ordinary, matched vertical and both XYZ modes released; vertical force stopped with no valid plan at 96.1°. All five timed out on scene 412. The benefit depends on starting geometry and optimizer seed.

On the four successful vertical-force runs the first useful rotation occurred at 5.50–7.30 s. Their mean strict release time was 8.30 s. Their mean axial contact location was **7.29 cm**, with **+8.88 N·m·s** signed hand impulse around support. The six failures under the same treatment averaged **2.08 cm** and **−1.67 N·m·s**, despite longer contact duty integrated over the episode (2.21 s versus 1.31 s). The two successful matched-XYZ runs averaged 12.90 cm and +12.13 N·m·s. These times and contact statistics describe different outcome subsets. They cannot establish a speed advantage. All 165 episodes had zero executed nongripper impulse and peak robot roll/pitch at most 11.61°. [Individual outcomes, times, contact measurements, saturation, validity and latency](exploration_trials.csv)

## Does changing the proposals help?

| Ordinary position MPC | Cone 82, 10 seeds | Diagnostic scenes 411/412 | Fresh 418/419/421 | Replication 443/444/445 | Fresh paired gains / losses vs local |
|---|---:|---:|---:|---:|---:|
| Local | 0/10 | 1/2 | 0/3 | 3/3 | reference |
| Four broad | 0/10 | 0/2 | 2/3 | 2/3 | 2 / 1 |
| Four joint | 1/10 | 0/2 | 3/3 | 3/3 | 3 / 0 |
| Sham | 0/1¹ | — | 2/3 | 3/3 | 2 / 0 |

¹ Targeted attribution on 82/104, the one ordinary joint release. With joint proposals, useful rotation started at 9.72 s and release completed at 11.8 s. The mean axial location was 8.48 cm with +8.36 N·m·s of tipping impulse. The same-seed sham timed out. On scene 421 the joint proposals also converted a sham timeout to a release. On 418 the local controller briefly reached 12.9° and lost the cone, so its failure includes a settling/release issue. Fresh ordinary joint versus sham had **one gain, zero losses and five shared successes**. The paired gain over the standard RNG stream is larger than the gain attributable to replacing the four draws.

| Strict releases on the same six fresh cones | Ordinary | XYZ force | Matched XYZ position |
|---|---:|---:|---:|
| Local | 3/6 | 3/6 | 4/6 |
| Four broad | 4/6 | 3/6 | 5/6 |
| Four joint | 6/6 | 3/6 | 3/6 |

Under joint sampling on 82, ordinary released 1/10 (104), XYZ force 2/10 (105, 108), and matched XYZ 0/10. On the fresh six the ordinary joint controller gained three releases and lost none against each XYZ controller. Mean final tilts including failures under joint sampling were 0.0°, 48.6° and 47.5° respectively. The six fresh scenes are too few for a population-wide ranking; the controller that solved all six also dropped the local controller's success on diagnostic scene 411.

At one-second intervals we restored the very same executed states from two broad ordinary runs and evaluated the four displaced local lanes, retaining the identical 12 shared proposals. On 82/101 none of the 48 broad or 48 displaced local candidates predicted a 10° gain; the best broad prediction across these checks gained 0.082°. On 418/131, after useful geometry had emerged, both sets predicted tipping: at 3 s their best gains were 82.3° and 85.8°. At 7 s all four in each set predicted at least 10°. The executed closed-loop comparisons show that sampling can change the approach and subsequent release. These sparse matched-state checks do not show a uniformly better distribution of tipping trajectories. [Matched native batches](exploration_batches.csv) · [Batch probe](../tests/exploration_batches.py)

## Tether transfer and latency

The rate planner reserved six of its existing 24 velocity proposals for uniform endpoints at a 1 s horizon. Box mass (3 kg), friction, slack, torso heading and 80 cm goal displacement matched the historical 65° scenario. Only goal bearing and paired CEM seed varied. Cable-crossing and direct-contact constraints stayed active.

| Bearing / CEM seed | Local final box error | Broad final box error | Outcome |
|---|---:|---:|---|
| 65° / 67 | 80.0 cm | 56.7 cm | Both stopped with 24 cable-crossing predictions; broad gained 24.7 cm along the task direction before stopping |
| 55° / 149 | 5.42 cm | 5.73 cm | Both timed out at 9 s, outside the 5 cm tolerance |
| 72° / 151 | 80.0 cm | 80.0 cm | Both stopped before moving the box |

Strict towing success was 0/3 for each sampler. No executed direct-contact impulse occurred. The partial transport at 65° does not justify promoting broad sampling for this task. [Tether comparison](../tests/tether_exploration.py)

| Complete planning update | Updates | p50 | p95 | Maximum | >100 ms |
|---|---:|---:|---:|---:|---:|
| Cone, pooled | 17,193 | 48.1 ms | 57.8 ms | 123.0 ms | 5 (0.029%) |
| Cone local | 8,461 | 49.0 ms | 58.4 ms | 103.5 ms | 1 |
| Cone broad | 2,919 | 48.4 ms | 57.8 ms | 78.6 ms | 0 |
| Cone joint | 5,240 | 46.8 ms | 56.9 ms | 123.0 ms | 4 |
| Cone sham | 573 | 45.7 ms | 55.3 ms | 71.7 ms | 0 |
| Tether, pooled | 277 | 61.4 ms | 71.7 ms | 79.9 ms | 0 |

These headless distributions include sampling, snapshot, all native steps, scoring, selection and any winner copies in the timed call. Production recording was added during the staged run; the first ordinary diagnostic runs used the earlier path-only recorder. Final compression, MPC-independent wrench replay and video encoding are outside the planner. Different episode mixtures and run times make small timing differences between sampler rows unsuitable for ranking their implementation speed. [Timing definitions, parity checks, video and plot instructions](episode_telemetry.md)

## Next experiment

Start with ordinary position MPC and keep local and joint samplers available. Compare the same initial physical state and fixed goal under receding-horizon MPC, execution of the original winning horizon without replanning, and a simple task baseline. Score prediction error at **matched** commands as a function of lookahead, in addition to strict closed-loop success. Replanning changes later commands; its future executed states alone are unsuitable as ground truth for an old unexecuted plan.

The current pool and live world share an immutable MuJoCo model. Both pool evaluation and world restoration verify the snapshot's model identity. Parameter mismatch needs an explicit transfer between topology-compatible models: verify joint, geom, actuator and tendon mappings; transfer physical state and real controller history/delay state; rebuild model-dependent derived data and decide how to initialize constraint warm starts. Preserve the current same-model bit-exact test, then use a separate identical-parameter, two-model control before perturbing friction or mass. Record both model identities, parameters, force schedules and privileged information available at plan time. The existing unexpected-pulse trace already separates modeled and surprise ticks. [Pool and model guard](../native/rollouts.h) · [World snapshot and restore](../native/world.h)

### Reproduction

From the repository root with the prepared native build, local scenes and policy:

```sh
export PYTHONPATH=tools:build:tests OPENBLAS_NUM_THREADS=1
py=.pixi/envs/default/bin/python
"$py" tests/exploration_trials.py principal off conditioned-z matched-z conditioned-xyz matched-xyz
"$py" tests/exploration_trials.py principal off --broad
"$py" tests/exploration_trials.py principal off conditioned-xyz matched-xyz --joint
"$py" tests/exploration_trials.py attribution off --sham
"$py" tests/exploration_trials.py secondary off conditioned-z matched-z conditioned-xyz matched-xyz
for sampler in --broad --joint; do
    "$py" tests/exploration_trials.py secondary off "$sampler"
done
for split in validation replication; do
    "$py" tests/exploration_trials.py "$split" off conditioned-xyz matched-xyz
    for sampler in --broad --joint; do
        "$py" tests/exploration_trials.py "$split" off conditioned-xyz matched-xyz "$sampler"
    done
    "$py" tests/exploration_trials.py "$split" off --sham
done
"$py" tests/exploration_metrics.py
"$py" tests/exploration_batches.py
"$py" tests/tether_exploration.py
```

The trial runner reuses completed recordings in `artifacts/exploration`. The command `pixi run view-cone --difficulty general --scenario-seed 419 --cem-seed 137 --broad-proposals` runs the corresponding interactive ordinary controller. Use `--joint-proposals` for the correlated sampler.
