# UniFP-MPC

The pretrained UniFP-Spot controller pushes a movable box to an XY goal in MuJoCo. A Judo-derived CEM planner selects world-space finger targets using parallel native rollouts of the policy. The first closed-loop demonstration pushed a 1 kg box toward an 8 cm goal and settled 5.8 mm from its target. [Recorded episode](artifacts/box_mpc/mpc/episode.mp4)

## Fixed-command force experiments

Paired native runs measure how virtual force inputs affect box contact, towing, payload sag and recovery from a physical finger disturbance. [Measurements, plots and reproduction commands](docs/fixed_policy_loads.md)

## Watch tethered box transportation

```bash
pixi run --locked view-tether
```

Spot tows a free box with a physical 70 cm cable. Ordinary velocity MPC settled four of five 80 cm goals, including a diagonal pull with a 76° change in loaded cable bearing. The viewer cycles reproducible starts and shows the winning box path and measured tension. [Experiments, videos and reproduction commands](docs/tether.md)

## Locomotion under physical torso loads

The three loaded-locomotion controllers each settled nine matched 90 cm walks. Virtual base force changed the actor at equal effective velocity targets. Ordinary velocity MPC had the lowest mean path error on the held-out conditions. [Task, measurements, videos and reproduction commands](docs/base_load.md)

## Watch measured wall-force MPC

```bash
pixi run --locked view-wall --padded --force 15 --distance -.45 --ee-force conditioned --scenario-seed 91 --cem-seed 67
```

The window displays the measured normal load, setpoint, base goal and winning EE paths. This run maintained contact through 99.3% of the traverse and finished 1.1 cm from the base goal. Walking RMS force error was 5.65 N. Omit `--distance` for stationary regulation. [Wall experiments, videos and reproduction commands](docs/wall_contact.md)

## Watch traffic cone uprighting

```bash
pixi run --locked view-cone
```

The viewer cycles through near, varied and general approaches. Green marks world upright. Cyan shows the raw EE request, dashed violet the projected target, and amber the predicted finger. A white dot marks its position at replanning. CEM optimizes all six commands with a soft 4 cm torso/leg clearance term. To watch Spot approach from a 72° offset, add `--difficulty general --scenario-seed 80 --cem-seed 43 --episodes 1`. To reproduce the original 6.2 s demonstration, use `--difficulty near --objective upright_release --episodes 1`. [Side-approach video](artifacts/cone_clearance/general-buffer/80/episode.mp4) · [Model, costs and matched trials](docs/cone_uprighting.md)

For the away-pointing cone demonstration, run

```bash
pixi run --locked view-cone --difficulty general --scenario-seed 82 --cem-seed 43 --objective plate_access --joint-proposals --episodes 1
```

This experimental controller released the cone at 10.0 s. The same scene timed out with CEM seed 67. [Success video](artifacts/ee_study/general-plate_access--joint-proposals/82/episode.mp4) · [Matched failure video](artifacts/ee_study/repeat67-plate_access/82/episode.mp4) · [EE tracking, SUMO comparison and experiments](docs/ee_tracking_cone.md)

To try jointly optimized EE force, run `pixi run --locked view-cone --difficulty general --scenario-seed 82 --cem-seed 67 --ee-force conditioned --force-axes xyz --episodes 1`. Use `--force-axes z` for vertical-only force or `--ee-force matched` for an equal-dimensional reachable position offset. [Cone force experiments and held-out comparisons](docs/cone_force.md)

## Watch live randomized box MPC

From the repository root in a graphical Linux session with the single-world artifacts prepared, run

```bash
pixi run --locked view-box
```

The window cycles through easy, medium, hard, oblique, extended, crossing and general pushes until closed. By default CEM jointly samples world-space EE position and body forward, lateral and yaw velocities. Drag to orbit and scroll to zoom. Cyan shows the winning raw EE requests, dashed violet the projected targets, amber the predicted finger through the full horizon, and green the 15 mm goal. The window counts settled episodes at 50, 15 and 10 mm. Seeds, friction, progress, settling times and planning latency are appended to `artifacts/live_box/episodes.jsonl`.

For a 2.02 m side approach and a 66 cm push, run `pixi run --locked view-box --difficulty general --scenario-seed 12 --cem-seed 17 --episodes 1`. Use `--seconds` to override the tier's control limit, `--pause 2` to keep the outcome visible, or `--headless` for controller timing. The 24-candidate, one-iteration, 0.5 s baseline and alternatives were compared on matched starts. [Expanded distributions and results](docs/general_box.md)

To replay the earlier forward-only hard approach, use `--base forward --difficulty hard --scenario-seed 2 --cem-seed 17 --episodes 1`. [Original viewer measurements](docs/live_box.md)

For a side approach with coordinated turning and sidestepping, run

```bash
pixi run --locked view-box --base stance --difficulty oblique --scenario-seed 0 --cem-seed 17 --episodes 1
```

Spot turned 41° before contact and settled a 41 cm push 3.1 mm from the goal in this episode. Stance guidance also recovered a long push from a saved slip state. Late corrections and settling on new side starts remain challenging. [Recovery experiments and latency](docs/contact_recovery.md)

Use `pixi run --locked view-box --base guided --difficulty oblique --scenario-seed 1 --cem-seed 17 --episodes 1` to watch CEM choose all three base rates alongside the world EE target. This episode settled 3.6 mm from the goal. The four-mode comparison includes held-out starts, failed recoveries and full planning latency. [Guided CEM results](docs/guided_cem.md)

To watch the 5 kg force-conditioned push, run `pixi run --locked view-box --difficulty easy --scenario-seed 4 --cem-seed 17 --box-mass 5 --ee-force conditioned --episodes 1`. Use `--ee-force matched` to optimize the equivalent reachable target with zero force input. [Paired box and wall experiments](docs/ee_force_box.md)

## Run a single world

Run these commands from this directory on Linux x86-64 with the local UniFP-Spot checkout available.

```bash
export PATH="$HOME/.pixi/bin:$PATH"
export PIXI_CACHE_DIR="$PWD/.pixi/cache"
pixi install --locked
pixi run --locked build
../unifp-spot/.venv/bin/python -B tools/export_policy.py
../unifp-spot/.venv/bin/python -B tools/export_contact.py
pixi run --locked python tools/prepare_scene.py
pixi run --locked env PYTHONPATH=build python tools/run_world.py push --save artifacts/scripted_push.npz
```

The runner also accepts `stand`, `walk` and `reach`. One persistent C++ `World::step` inserts the current MuJoCo observation and command into history, evaluates the ONNX encoder and actor, and applies delayed torques over four physics steps. Python supplies 12-column command schedules and receives the executed 50 Hz trace. Velocities use m/s and rad/s, EE targets use (radius, elevation, azimuth), and EE and base force commands use N in the yaw frame. Consecutive calls to `World.run` continue from the last executed tick. In the trace, state is post-step, action is the one just applied, and forces are means over those same four substeps. [Controller step](native/world.h) · [Actuators](native/plant.h) · [Command examples](tools/run_world.py)

```bash
pixi run --locked env PYTHONPATH=build python tests/check_actuators.py
pixi run --locked env PYTHONPATH=build:tools python tests/check_world.py
```

The project-local package cache is writable in this workspace. Pixi 0.84.1 accepts the v6 lock with `--locked`. The `--offline` option also works with the populated cache here. The build task runs CMake in Release mode with Ninja. MuJoCo headers and library come from the same wheel, and ORT headers and library come from one CPU development package. The one-time exports use the source environment for Torch and USD. The native runtime loads two local artifacts, the ONNX policy and a compiled MuJoCo scene. Scene preparation takes the local Judo checkout, Menagerie Spot mesh directory and extracted UniFP USD gripper geometry. Paths can be changed with `tools/prepare_scene.py --help`. The measurement scripts require the archived recordings in UniFP-Spot. To reproduce the export environment on another machine, UniFP-Spot supplies its own Linux uv lock. An out-of-tree environment can be created with `UV_PROJECT_ENVIRONMENT="$PWD/.export-venv" uv sync --frozen --no-install-project --project ../unifp-spot`, then used with `.export-venv/bin/python -B tools/export_policy.py`. The tested export used the already installed source environment. [Runtime manifest](pixi.toml) · [Runtime lock](pixi.lock) · [Source environment](../unifp-spot/pyproject.toml)

The exporter imports the tracked `ActorCritic` implementation and strictly loads all weights before extracting the deterministic encoder and actor. It checks the graph and numerical parity before writing `artifacts/encoder_actor.onnx`. The replay probe also needs the local Isaac recordings under `outputs/validation/free_space_x_reset_verified` and `outputs/mpc/latency_cpu_constant_writes_seed0`, the Judo checkout, and Menagerie's `boston_dynamics_spot` assets. Use `--source`, `--judo` and `--menagerie-spot` on the probe for other locations. The recordings and local MPC probes are outside UniFP-Spot's tracked source at the inspected revision. The graph export also works independently of those recordings. [Exporter](tools/export_policy.py) · [Replay check](tests/check_replay.py) · [Native check](tests/native_probe.cpp)

### Previously verified inference and kinematics

| Check run | Result |
|---|---|
| Checkpoint identity | Iteration 5300, SHA-256 `81ba5bafc4535c4e457cc38bb2b8f882e2b54e0a892b89466c81f502bf8f6106` |
| Export | Opset 18, float32 `history[B,2528] → action[B,19]`, eight affine layers and six ELUs. SHA-256 `4b1f0b65467b3f85cef03f97a77f79988da138cc627d2219f909a87100a2c060` matches the archived Isaac predictor graph byte for byte |
| ORT 1.26.0 vs PyTorch `act_inference` | Maximum absolute raw action difference 1.19e-6 on a zero-padded synthetic history and 2.86e-6 on a batch of eight |
| Observation and target reconstruction | All 79 entries over 300 walk frames and 300 free-space +20 N command frames, maximum absolute observation difference 7.06e-6. Recorded joint targets passed the default-plus-scaled-action check. Five histories per trace, including the first two actions after reset, reproduced executed actions to 9.54e-7 |
| Native ORT 1.24.2 vs executed Isaac actions | All 69 complete history windows per 100-tick trace. Maximum absolute error 1.91e-6 for reach and 1.43e-6 for the archived wall diagnostic. Batch one also passed |
| MuJoCo FK vs Isaac | Set root pose and all 19 named joints from each of 200 executed poses. Maximum distance between `arm_link_fngr` body origins 7.50e-7 m |

### Single-world transfer results

The comparison uses archived seed-0 PhysX runs and the same commands. The available host ran MuJoCo on CPU and had no accessible NVIDIA driver for a fresh Isaac run. Free-space cases ran for six simulated seconds each. The table compares samples from 3.76–5.98 s. Reach error is the mean 3-D distance from the finger origin to the target in the base-yaw frame.

| Closed-loop measurement | MuJoCo | Isaac |
|---|---:|---:|
| Stand, mean base height | 0.518 m | 0.518 m |
| Stand, mean planar speed / maximum absolute roll or pitch | 0.0178 m/s / 7.35° | 0.0151 m/s / 7.71° |
| Walk, mean forward speed for 0.30 m/s commanded | 0.2987 m/s | 0.2942 m/s |
| Reach, mean target error / lateral displacement relative to stand | 5.17 cm / +11.62 cm | 5.02 cm / +11.67 cm |
| Free-space +20 N forward EE input, forward shift relative to stand | +3.51 cm | +3.62 cm |
| Matched +10 cm position-only input, forward shift relative to stand | +5.65 cm | +5.76 cm |

At identical recorded joint states and targets the native drive reproduced all 152 applied torque samples exactly. Signed saturation tests at zero and three ticks of delay, including knee-table knots, midpoints and out-of-range angles, had a maximum error of 3.82e-6 Nm. Driving the floating MuJoCo robot with the archived targets for eight substeps gave 0.00610 rad RMS and 0.0253 rad maximum joint-position error. The ground and finger-stop response times were set to 10 ms using this reset transient. Beyond that calibration window, open-loop playback of the standing joint targets gave 0.00730 rad RMS and 0.0349 rad maximum error at the 60–200 ms policy ticks. The walking, reaching and box comparisons use separate trajectories. [Actuator check](tests/check_actuators.py) · [Isaac reset recording](../unifp-spot/outputs/validation/reset_verified/stand_0_reset.npz) · [Free-space recordings](../unifp-spot/outputs/validation/free_space_x_reset_verified/report.json)

For the push, the SUMO/Judo free-box scene uses the Isaac object's 1 kg mass, 0.40 × 0.40 × 0.50 m dimensions, 0.30 sliding friction and folded arm start. Eight USD convex hulls model the finger, fixed jaw and wrist. The robot holds its EE target at (0.20, 0, 0.10) m for 1 s, then commands (0.48, 0, −0.20) m. Base velocity and force inputs are zero. The force-enabled comparison uses x = 0.38 m and +20 N, giving the same effective position target. Values below cover t = 1.00 through 9.98 s. First contact is reported on the 20 ms clock when gripper force exceeds 1 N in a substep.

| Script | Box Δx, Δy (cm) | First contact | Peak / mean world +x gripper force on box | Substeps above 1 N |
|---|---:|---:|---:|---:|
| MuJoCo, hold warmup target | 0.00, 0.00 | None | 0 / 0 N | 0% |
| MuJoCo, nominal x = 0.38 m, zero force | +5.14, −1.41 | 3.50 s | 12.88 / 0.71 N | 11.0% |
| MuJoCo, matched position only | +7.11, −1.79 | 1.84 s | 20.78 / 0.60 N | 8.1% |
| Isaac, matched position only | +8.50, −0.11 | 1.56 s | 32.07 / 0.86 N | 14.2% |
| MuJoCo, +20 N input | +6.68, −3.60 | 1.88 s | 17.84 / 0.58 N | 8.8% |
| Isaac, +20 N input | +7.05, +1.17 | 1.56 s | 30.36 / 0.94 N | 17.1% |

Both engines moved the box through physical gripper contact. In MuJoCo the nongripper robot–box impulse was zero in all four trials. The world-frame gripper and ground impulses plus gravity matched the box's measured change in momentum to within 2e-16 N s per component per policy tick. Splitting the 500-tick push into two calls after tick 94 reproduced every returned field exactly. [World check](tests/check_world.py) · [Isaac position run](../unifp-spot/outputs/playground/box_1kg_position_matched/run.json) · [Isaac force-input run](../unifp-spot/outputs/playground/box_1kg_20N/run.json)

These controlled runs use fixed archived actuator delays and deterministic physics. The gripper and joint stops use the source USD, while the rest of the robot retains Judo's collision approximations and exclusions. MuJoCo uses compliant contacts, Newton and implicitfast. PhysX uses TGS with its own friction, depenetration and velocity limits. The recorded 100 rad/s joint velocity ceiling has no explicit MuJoCo counterpart here. Contact started later in MuJoCo, produced lower peaks and led to different lateral slip. Those differences matter for scoring short pushing trajectories. Cross-engine validation of force regulation and object-goal accuracy remains open.

## Push a box to an XY goal

After building the runtime and preparing the scene above, run the planner from the repository root. The goal is the absolute box-center position in world metres. The default center starts at (1.08, 0.00).

```bash
pixi run --locked env OPENBLAS_NUM_THREADS=1 PYTHONPATH=build python tools/box_mpc.py --goal 1.16 0 --out artifacts/box_mpc/mpc
pixi run --locked env OPENBLAS_NUM_THREADS=1 PYTHONPATH=build:tools python tests/check_box_mpc.py
pixi run --locked env MUJOCO_GL=egl python tools/render_box.py artifacts/box_mpc/mpc
```

The comparison runs the hold, scripted and MPC controllers from the same contact-free, one-second warmup. Each run writes the executed 50 Hz trace and a JSON summary. MPC also saves the selected knots, executed requests and timing for every replan. Rendering uses the saved trace and requires FFmpeg with H.264 and subtitles on the path, or an executable supplied with `--ffmpeg`. [Controller](tools/box_mpc.py) · [Comparison](tests/check_box_mpc.py) · [Renderer](tools/render_box.py)

Every 0.1 simulated seconds the planner takes one live snapshot. It samples 32 trajectories with three linear position knots over 0.5 s, limits requested EE speed to 0.5 m/s and keeps base and force commands at zero. Eight workers predict through the shared UniFP controller at 50 Hz with 200 Hz physics. CEM fits the two best valid candidates. The best evaluated trajectory supplies the next five live commands. The shifted winner and elite mean are both evaluated in the next batch. All 29 selected prefixes exactly reproduced the predicted poses, actions, projected policy commands and gripper forces. [CEM adaptation](tools/cem.py) · [Native evaluator](native/rollouts.h)

The cost is the squared terminal box-center XY error plus 0.05 times the squared terminal 3D distance from the finger to the pushing face. At each replan the desired push direction is the live box-to-goal vector. A ray opposite that vector intersects the predicted box footprint to locate the approach point, 0.125 m above the center. All 32 candidates in the initial update had identical 8 cm box errors and zero gripper contact, so the finger term supplies the early ranking. Inside the 15 mm goal radius only box error is scored. Invalid forecasts are ineligible. Success requires 0.5 s continuously inside the radius and a box XY speed at or below 2 cm/s throughout the final 0.1 s.

| Controller | Box displacement X / Y (cm) | Final XY error (cm) | Max absolute base roll or pitch | Simulated / wall time (s) | Success |
|---|---:|---:|---:|---:|---|
| Hold initial EE request | 0.00 / 0.00 | 8.00 | 8.59° | 10.0 / 0.28 | No |
| Existing scripted push | +7.11 / −1.79 | 2.00 | 8.79° | 10.0 / 0.28 | No |
| MPC, 32 samples × 1 iteration | +7.57 / +0.39 | 0.58 | 7.84° | 3.9 / 1.57 | Yes |

At the common 3.9 s timestamp, hold, scripted and MPC were 8.00, 5.13 and 0.58 cm from goal. MPC stayed in tolerance for 0.56 s. On the eight-vCPU Xeon, full planning latency was 46.4 / 54.8 / 56.8 ms (median / p95 / maximum). The synchronous cycle including live execution took 48.3 / 57.0 / 58.7 ms, with 0/29 misses of the 100 ms deadline. The loop ran unpaced at 20.4 wall-clock updates/s and replanned at 10 Hz in simulation. Episode wall time includes setup and warmup and excludes saving and rendering. [Measured run](artifacts/box_mpc/mpc/summary.json)

Short additional trials reached an oblique (+5, +2) cm goal with 7.7 mm error, a 12 cm forward goal with 7.7 mm error, and the 8 cm goal from a 4 cm farther start with 12.4 mm error. A second sampling seed finished at 14.7 mm. At equal sample budget, 16 candidates × 2 iterations finished at 0.8 mm on the primary start.

### Coordinating the base and arm

Use `--controller mobile` to optimize one body-forward speed along with the world-space finger target. The default `mpc` controller runs the original position-only experiment. Rebuild the native extension before using four-column schedules.

```bash
pixi run --locked build
pixi run --locked env OPENBLAS_NUM_THREADS=1 PYTHONPATH=build python tools/box_mpc.py --controller mobile --goal 1.43 0 --out artifacts/mobile_box/mobile_long
pixi run --locked env OPENBLAS_NUM_THREADS=1 PYTHONPATH=build:tools python tests/check_mobile_box.py
```

On a 35 cm forward goal, the original controller hit UniFP's 0.55 m radial limit on 86% of executed ticks, including every tick of the last second. Its furthest requested X was 1.119 m against a 1.280 m world upper bound. It lost gripper contact for up to 2.44 s between pushes. On a pure lateral goal the box moved forward as well as sideways and overshot. The robot still made contact but failed to correct the error.

The mobile candidate adds one CEM variable to the six free EE knot coordinates. Body-forward speed ramps from the last executed request to a sampled value at 0.25 s, then holds across the rest of the 0.5 s horizon. The limits are ±0.35 m/s and 0.8 m/s². Lateral velocity, yaw rate and force inputs stay zero. Both controllers use the same arm bounds, projection, 32 candidates, two elites, one CEM iteration and eight native workers. [Sampler](tools/cem.py) · [Judo CEM](../judo/judo/optimizers/cem.py) · [SUMO Spot planner configuration](../sumo/sumo/controller/optimizer_overrides.py)

Adding speed to the original cost overshot the long goal and produced 0.088 N s of nongripper impulse. Mobile scoring rejects any predicted nongripper impulse and adds the squared shortfall below a 1.0 m terminal body-forward base-to-box separation to reserve working room. These changes alone left a 51.1 mm goal error. To limit lateral drift during a fast push, the terminal box target lies 8 cm ahead of the live box's projected progress along the start-to-goal line, capped at the final goal. Finger approach uses the same target. The seed-0 box's maximum lateral excursion fell from 5.0 cm to 0.54 cm. Stopping uses the actual final goal and the original settling criteria. [Cost and execution](tools/box_mpc.py)

| Goal from start | Position-only final error / control time | Mobile final error / control time | Mobile base ΔX / ΔY | Full cycle p95, position-only / mobile |
|---|---:|---:|---:|---:|
| +8 cm X | 5.8 mm / 2.9 s | 0.4 mm / 2.3 s | +15.4 / −3.1 cm | 54.6 / 55.8 ms |
| +35 cm X | 191.0 mm / 9.0 s limit | 3.0 mm / 3.7 s | +33.4 / −5.1 cm | 55.5 / 54.1 ms |
| +8 cm Y | 40.9 mm / 9.0 s limit | 5.4 mm / 3.5 s | +7.2 / −6.4 cm | 55.5 / 54.5 ms |
| +25 cm X, −4 cm Y from a (+2, +8) cm shifted box | 131.0 mm / 9.0 s limit | 10.9 mm / 3.3 s | +24.6 / −6.9 cm | 54.1 / 60.7 ms |

The paired runs used identical warmup states and seed 0. All stayed upright (maximum absolute roll or pitch 8.2°). The mobile long run took 4.7 simulated seconds including warmup, 1.97 s wall time. Full planning took 49.3 / 52.3 / 57.1 ms (median / p95 / maximum). Including live execution the cycle took 51.2 / 54.1 / 59.4 ms with 0/37 misses of 100 ms. Every executed prefix exactly reproduced its forecast. Seed 1 on the same long goal took 7.4 s of control and finished at 13.4 mm. [Measurements](artifacts/mobile_box/mobile_long/summary.json) · [Episode video](artifacts/mobile_box/mobile_long/episode.mp4)

With the mobile objective and speed constrained to zero, closed-loop MPC moved the box 16.2 cm and missed the long goal by 18.8 cm. Replaying only the winning arm requests moved it 8.6 cm forward. Replaying only the winning base requests with the initial base-relative EE command advanced the base 37.3 cm with zero gripper contact and zero box motion. Replaying both reproduced every live trace field, with 35.16 cm forward box travel, 0.26 cm lateral drift, 77 gripper-contact ticks and zero nongripper impulse. [Matched trials and channel replays](tests/check_mobile_box.py)

On the short forward goal, position-only MPC also succeeded with a (+2, +8) cm box offset (13.2 mm error), box sliding friction of 0.50 versus 0.30 (8.4 mm), and a warmup EE request shifted 9 cm sideways (10.6 mm). Clearing the CEM mean, incumbent and variance at each replan on seed 0 gave 13.3 mm in 36 updates versus the warm-started 5.8 mm in 29. The world-space EE search box remains fixed throughout the episode. The randomized study below also tests bounds that translate with the box. Contact transfer to PhysX remains open.

### Randomized starts and contact friction

For randomized forward and oblique pushes, the lower-cost configuration uses `--controller mobile --objective bounded --candidates 24`. It settled three of four held-out tasks, the same count as the 32-candidate reference. Mean final error fell from 25.8 to 17.6 mm and full planning p95 from 52.2 to 39.7 ms. On the original nominal 35 cm demonstration it missed the 15 mm tolerance. Use the default mobile settings to reproduce the archived 8, 35 and 50 cm demonstrations. [Experiments, limits and reproduction commands](docs/box_robustness.md) · [Per-episode measurements](docs/box_trials.csv)

```bash
pixi run --locked python tools/box_scenario.py --seed 12 --out artifacts/robust_box/scenes/12
pixi run --locked env OPENBLAS_NUM_THREADS=1 PYTHONPATH=build python tools/box_mpc.py --controller mobile --base forward --objective bounded --candidates 24 --seed 29 --scenario artifacts/robust_box/scenes/12 --out artifacts/example_random_push
```

## Evaluate a rollout batch

Build and prepare the local artifacts as above. With `PYTHONPATH=build:tools`, this example starts two predictions at the same instant during gripper–box contact.

```python
import numpy as np
from run_world import command_schedule
from unifp_native import Rollouts, World

live = World("artifacts/spot_box.mjb", "artifacts/encoder_actor.onnx")
pool = Rollouts(live, workers=4)
live.run(command_schedule("push")[:93])
executed = live.snapshot()
targets = np.empty((2, 25, 3), dtype=np.float32)
targets[0] = [.825, -.04, .444]
targets[1] = [.825, -.02, .444]
predicted = pool.evaluate(executed, targets)
box_final_xyz = predicted["qpos"][predicted["valid"], -1, -7:-4]
```

Pass interpolated world EE positions in metres as `[candidates, ticks, 3]`. For coordinated motion, append body-forward velocity in m/s as column four. Column five adds body-lateral velocity in m/s. Column six adds yaw rate in rad/s. At each tick the controller projects XYZ using that candidate's pre-action base position and yaw. The three-column form supplies zero velocity. Widths 3–6 supply zero force. Width 8 appends world-frame horizontal EE force in N. Width 9 appends three EE force components in the instantaneous gravity-aligned body-yaw frame, also in N. Pass `matched_position=True` to both evaluate and run to issue the corresponding reachable position with zero force. The planner evaluates its three knots at `np.arange(25) * .02`. After selection, `live.run(targets[selected, :5])` projects and executes the same 100 ms prefix. Direct twelve-channel policy schedules also work. A single C++ call covers the whole batch. Treat input arrays as read-only until it returns. [Planner interpolation and constraints](../unifp-spot/tests/probe_position_mpc.py) · [Native command conversion](native/world.h)

`evaluate` returns owned NumPy arrays with candidate and tick as their first two axes. `qpos` and `qvel` use native MuJoCo order. In the prepared box scene `qpos[..., :7]` is the base pose and `qpos[..., -7:]` is the box pose, both xyz followed by a wxyz quaternion. The trace also includes post-step `ee_world`, the applied `policy_command` and `action`, mean signed world-frame gripper and ground forces on the box, peak forward gripper force, contact duty above 1 N and nongripper robot–box impulse. Each candidate has a `valid` flag, `valid_steps` count and `failure` string. Numeric or physical failures, including excessive tilt, leave the failed tick and remaining rows filled with NaNs. Python can use these arrays directly to evaluate box progress, approach, stability and contact.

Capture a snapshot at a policy-tick boundary. Reuse it across all CEM iterations until the live world advances. The snapshot owns the MuJoCo integration state (including time, object state and solver warmstart), warning counters, previous action, all 32 observation frames, gait tick, delayed target ring and index, and last-contact measurements. Fixed gains and lags come from the shared read-only model. Snapshots are process-local and associated with their originating world. Each worker has its own `mjData`, policy tensors, controller state and ORT run options. The pool shares the live model and CPU ONNX session. It uses one ORT thread per inference and one native worker per active lane, including the caller. Extra threads sleep between batches. [Snapshots and shared step](native/world.h) · [Persistent workers](native/rollouts.h)

### Validation and timing

```bash
pixi run --locked env PYTHONPATH=build:tools python tests/check_branching.py
pixi run --locked env OPENBLAS_NUM_THREADS=1 PYTHONPATH=build:tools python tests/benchmark_rollouts.py --csv artifacts/rollout_benchmark.csv
```

Serial restoration reproduced uninterrupted actions, joint and box states, EE motion and contact exactly at reset, after seven ticks and at tick 93 during the push. Repeated and reordered candidates, successive planning iterations, moving the live world forward, reusing an older snapshot and recovery from a failed candidate all passed. Four workers passed the same tests, and the benchmark compared all ten trace arrays across one, four and eight workers on 32 scheduled candidates. The nine original live trace fields matched the saved 500-tick single-world push exactly. A separate check passed eight candidate splines from the existing Isaac planner's sampling, constraint and interpolation functions into this backend. Its projected commands matched the source PyTorch `position_command` at all 200 predicted pre-action poses to 1.20e-7 maximum absolute error. [Branch tests](tests/check_branching.py) · [Benchmark](tests/benchmark_rollouts.py)

The benchmark ran on an eight-vCPU Xeon w3-2435 VM with MuJoCo 3.6.0 and ONNX Runtime CPU 1.24.2. Each cell contains 21 warm calls from the same contact snapshot. Each candidate covers 25 policy ticks and 100 physics steps (0.50 simulated seconds). All 32 candidates made measured gripper contact and moved the box forward by 1.04–2.76 cm. The timer includes a fresh live snapshot, allocating and filling the returned arrays, worker dispatch, projection, inference, physics, and return to Python. Candidate generation, CEM updates, cost evaluation and live execution fall outside the timer.

| Candidates | 1 worker p50 / p95, ms | 4 workers p50 / p95, ms | 8 workers p50 / p95, ms |
|---:|---:|---:|---:|
| 1 | 8.5 / 9.7 | 8.2 / 8.4 | 8.2 / 9.1 |
| 8 | 65.7 / 68.5 | 17.4 / 19.0 | 14.7 / 18.4 |
| 16 | 131.7 / 132.8 | 34.1 / 35.9 | 24.1 / 28.4 |
| 32 | 261.6 / 266.3 | 68.4 / 71.0 | 48.3 / 53.3 |

With eight workers and 32 candidates the slowest batch took 54.7 ms, with 0/21 misses of 100 ms and a 5.42× median speedup. Serial missed all 21 deadlines for both 16 and 32 candidates. Mean per-candidate worker elapsed time at 32 candidates was 4.80 ms in inference and 3.33 ms in physics/contact for one worker, and 5.62 ms and 4.12 ms respectively for eight. These component times overlap across workers. Eight workers used 5.87 CPU cores on average, or 73% of available capacity. Median live snapshot time was 0.015 ms and native batch time outside the longest worker's interval was 0.041 ms. The original MuJoCo thread-pool trial consumed about three idle cores with four lanes configured. The condition-variable workers consumed under 0.001 cores in the same 250 ms idle check. [Raw timing samples](artifacts/rollout_benchmark.csv)

The closed-loop box experiment above measures the full cycle with sampling, scoring, CEM and live execution included. The single-world contact timing and lateral-slip differences above still apply to transfer across physics engines.

## Source and dependency audit

| Checkout under `/home/nexus` | Implementation and decision |
|---|---|
| `unifp-spot`, `dd97c6aa` | `ActorCritic.act_student` and `OnPolicyRunner._push_obs/_stack_obs` define the deployed network and history. `SpotPosForceCfg`, `UniFPEnv` and the observation/reward terms define the physical interface. The checkpoint lives at `pretrained/spot_unifp/model_5300.pt`. |
| `unifp-spot/tests`, local research files | `probe_position_mpc.py` implements `sample_cem`, `fit_cem`, `constrain_knots`, `position_command`, `rollout` and live selection in `run_controller`. This is a NumPy adaptation of Judo CEM with attribution. `probe_branching.py` holds `snapshot/restore`. `cpu_actuators.cpp` implements delay rings, PD and torque limits. `profile_position_mpc.py` is the timing entry point. Keep the executable PhysX baseline here and use these files as behavioral oracles. |
| `judo`, `96d6f451` | `CrossEntropyMethod.sample_control_knots/update_nominal_knots` and `controller/controller.py` supply the sampling, spline and update references. `HierarchicalMJSimulation` and `HierarchicalMJRolloutBackend` use the Spot extension. Both copies of `spot_locomotion.onnx` inspect as `[1,84] → [1,12]`. In `System::setObservation/policyInference` the 25 command entries drive torso, arm and leg requests, the network supplies 12 leg outputs scaled by 0.2, and the arm is commanded directly. `SystemUtils::setState` restores qpos/qvel. `threadedRollout` starts and joins threads per call; systems can share an ORT session. |
| `sumo`, `4cea4e4e`; `sumo/.judo-src`, `a2da9f51` | SUMO locks Judo's `dta/fix_for_sumo` revision, distinct from standalone Judo. `make_controller` delegates to Judo and registers a G1 backend. Spot inherits the Judo path. `SpotAssetMixin` and `SpotBase` resolve Menagerie meshes and task XML includes. `g1_extensions/rollout.cpp` demonstrates a persistent pool. `SpotBoxPush.reward` combines object-goal distance, gripper proximity and object velocity. `SpotConeUpright.success` checks object +z against world +z with dot product ≥0.9. |
| `unifp-mpc` | The shared UniFP physics step, policy and rollout workers live in `native/`. Scene preparation, export and the headless runner live in `tools/`. Generated models and traces stay local in `artifacts/`. |

[Actor](../unifp-spot/unifp_spot/learning/ppo_cse_pf/actor_critic.py) · [History](../unifp-spot/unifp_spot/learning/ppo_cse_pf/on_policy_runner.py) · [Isaac MPC](../unifp-spot/tests/probe_position_mpc.py) · [Actuator kernel](../unifp-spot/tests/cpu_actuators.cpp) · [Judo CEM](../judo/judo/optimizers/cem.py) · [SUMO's pinned native Spot loop](../sumo/.judo-src/mujoco_extensions/system/system_class.cpp) · [Box](../sumo/sumo/tasks/spot/spot_box_push.py) · [Cone](../sumo/sumo/tasks/spot/spot_cone_upright.py)

**Environment split.** The archived training environment lists Python 3.11.13, Isaac Sim 5.1.0 and Torch 2.7.0+cu128. The local reference checkout requires Python 3.12 and IsaacLab 3.0.0. Its installed environment has Python 3.12.14, Isaac Sim 6.2.0.0, Torch 2.11.0+cu130, NumPy 2.5.4, ONNX 1.23.2, ORT 1.26.0 and MuJoCo 3.12.0. Both Judo and SUMO require Python ≥3.10 and MuJoCo ≥3.6,<3.7. Judo's Pixi setup pins native ORT 1.19.x and Linux GCC 13. SUMO's build requests ORT ≥1.22,<2, with Python and C++ ORT 1.24.2 installed. Their original extensions require C++17, CMake, pybind11, Eigen and OpenMP. The new 65-package Linux lock uses Python 3.13.12, NumPy 2.4.2, MuJoCo 3.6.0, native ORT 1.24.2 CPU, GCC/G++ 14.3.0, CMake 4.2.3, Ninja 1.13.2 and pybind11 3.0.2. It was installed into a fresh local prefix with Pixi 0.84.1 using cached packages. [Training freeze](../unifp-spot/docs/environments/cluster_a40_isaacsim-5.1.0.txt) · [UniFP lock](../unifp-spot/uv.lock) · [Judo build](../judo/mujoco_extensions/CMakeLists.txt) · [SUMO manifest](../sumo/pyproject.toml)

**Reuse and licenses.** Judo and SUMO include MIT licenses, with notices required for copied portions. The cached Menagerie Spot directory at revision `bf75643` carries Clearpath's BSD-3-Clause notice. UniFP-Spot lacks a top-level license and attributes its vendored USD and actuator configuration to RAI. Use local references for the robot and weights. The Judo primitive box needs no external object mesh. The transfer diagnostic uses the 1 kg Isaac box dimensions and mass. SUMO also has a distinct 20 kg rugged-box task. Its 3.36 kg cone uses a base, feet and stacked collision primitives plus a separate visual mesh. Preserve component notices and establish mesh redistribution terms before bundling visuals. [UniFP provenance](../unifp-spot/README.md) · [Judo license](../judo/LICENSE) · [SUMO license](../sumo/LICENSE) · [Box scene](../judo/judo/models/xml/spot_box/robot.xml) · [Cone object](../sumo/sumo/models/xml/objects/traffic_cone/traffic_cone.xml)

## Policy and model contract

One 20 ms policy tick covers four 5 ms physics steps. History is float32, oldest first, with zero padding at reset. Insert the current observation once, after setting the candidate command and before inference. Actor input is the newest 79 entries followed by the 64-value encoded history. The decoder estimates diagnostic quantities including synthetic external disturbances. [Environment](../unifp-spot/unifp_spot/envs/unifp_env.py) · [Observation definitions](../unifp-spot/unifp_spot/envs/base/mdp/observations.py)

| Frame slice | Quantity and scale |
|---|---|
| 0–1 | Base roll and pitch in radians |
| 2–4 | Body angular velocity ×0.25 |
| 5–23 | Joint position minus the task's default angle |
| 24–42 | Joint velocity ×0.05 |
| 43–61 | Previous applied policy action |
| 62–63 | Sine and cosine of the episode gait clock, period 0.64 s |
| 64–78 | Body velocity request (vx,vy) ×2, yaw rate ×0.25, EE (radius, elevation, azimuth), three zero orientation entries, yaw-frame EE and base force requests ×0.01 |

The runner sanitizes and clips frames to ±100. The environment sanitizes and clips actions to ±100 and targets `q_default + 0.25 * action`. XY velocity ranges are ±0.6/0.4 m/s, yaw rate ±0.6 rad/s. The EE sphere is centered at base position plus the yaw-rotated offset (0.35, 0, 0.12) m. Radius is 0.15–0.55 m, elevation −40° to +50°, azimuth ±90°. EE/base virtual command ranges are ±40/30 N per component. Only the independently scheduled external disturbance stream physically applies a wrench. The EE tracking reward offsets its Cartesian target by (commanded + external force)/200. Plan world-space targets and convert and project against each candidate's current base at each tick. [Spot overrides](../unifp-spot/unifp_spot/envs/spot/spot_pos_force_cfg.py) · [Reward](../unifp-spot/unifp_spot/envs/base/mdp/rewards.py) · [Projection](../unifp-spot/tests/probe_position_mpc.py)

Policy joint order in the recorded articulation is:

```text
arm_sh0 fl_hx fr_hx hl_hx hr_hx   arm_sh1 fl_hy fr_hy hl_hy hr_hy
arm_el0 fl_kn fr_kn hl_kn hr_kn   arm_el1 arm_wr0 arm_wr1 arm_f1x
```

MuJoCo's Judo model groups four complete legs before the arm. Resolve qpos, dof and actuator transmission indices independently by name. IsaacLab 3's root quaternion is xyzw and MuJoCo's is wxyz. Use the body frame for angular velocity and the yaw-only frame for the spherical task. The trained stance has front/hind hip pitch 0.9/1.1 rad, knees −1.5 rad, hip roll ±0.1 rad and initial base height 0.5 m. The task overrides the asset defaults and the leg gains. [Runtime articulation record](../unifp-spot/outputs/validation/free_space_x_reset_verified/report.json) · [MJCF actuator order](../judo/judo/models/xml/spot_primitive/actuator.xml)

**Model choice.** Start from Judo's Spot kinematics. At 20 positive-mass links, the maximum USD-to-compiled-MJCF differences were 9.32e-7 kg in mass, 3.86e-9 m in a local CoM coordinate and 1.77e-8 kg m² in a link-frame inertia tensor component. These compare authored nominal USD properties, with principal-axis rotations applied. The composed USD has 19 mesh, four sphere and two cube collision shapes. Judo uses boxes, capsules and spheres, including jaw approximations and explicit self-contact exclusions. The ground friction settings differ (Isaac static/dynamic 1.0 with average combine, Judo plane sliding coefficient 0.7). PhysX uses TGS and articulation iterations 4/0; Judo configures Newton/implicitfast at 10 ms. The single-world transfer uses the recorded passive parameters and the driven reset check above. [USD](../unifp-spot/unifp_spot/assets/spot/usd/spot_with_arm.usd) · [Actuation](../unifp-spot/unifp_spot/assets/spot/spot.py) · [MJCF defaults](../judo/judo/models/xml/spot_primitive/default.xml) · [Contact exclusions](../judo/judo/models/xml/spot_primitive/contact.xml)

Replace Judo's position servos with explicit torques. The task uses leg kp/kd 120/3, 45 Nm hips, joint-specific arm gains and effort limits, and knee torque clipping interpolated by measured angle. Delay lags span 0–3 physics ticks. The reference kernel delays position, velocity and feedforward targets and evaluates PD at each substep. The PhysX solver effort cap is 1e9 so the actuator performs clipping. Judo's legs use 60/1.5, knees a fixed ±115 Nm, and shoulder pitch armature is 0.7181504 versus the Isaac actuator configuration's 0.01. The archived articulation buffers give zero solver drive stiffness, damping and static, dynamic and viscous joint friction. Armature is zero for the legs, 0.01 kg m² for the six arm joints and 0.001 kg m² for the finger. The native drive updates one position-target ring because this policy supplies zero target velocities and feedforward efforts. At the recorded reset, group delays in 5 ms ticks are hip 0, knee 2, and (sh0, sh1, el0, el1, wr0, wr1, f1x) = (2, 0, 3, 0, 1, 1, 3). [Kernel](../unifp-spot/tests/cpu_actuators.cpp) · [Controlled reset](../unifp-spot/tests/validation_simulation.py) · [Articulation buffers](../unifp-spot/outputs/mpc/batch_reach_restore_seed0/executed_snapshot.pt)

## Planner decision and acceptance gates

Keep the existing NumPy CEM adaptation, linear interpolation and evaluated-best selection in Python. The native batch interface returns measured trajectories and validity for Python cost evaluation. SUMO's G1 pool and Judo's shared ORT session informed the worker design. The UniFP workers preserve the same policy and actuator step used in live execution.

The inspected PhysX GPU pilot spent 1.102/1.525 s mean planning for reach/wall with eight parallel samples and a dependent elite-mean pass. The published host regions put physics at 432/744 ms and encoder+actor at 53/53 ms across the two passes. The later one-pass CPU PhysX predictor with GPU execution, C++ actuation and ONNX took 136/189 ms mean and 145/198 ms p95 for planning. All 40 updates missed 100 ms. These historical workloads differ in placement and number of forecasts. A native box speedup requires a new same-task measurement. [Archived timing analysis](../unifp-spot/docs/mpc_latency.md) · [CPU run data](../unifp-spot/outputs/mpc/latency_cpu_constant_writes_seed0/latency_comparison.json)

| Stage | Smallest experiment and gate |
|---|---|
| M0 and early M1 | Locked native build, strict checkpoint load, executed-input replay and static FK passed. |
| M1, single world | The controlled reset torque replay, driven joint response, standing, walking and free-space reach checks passed on the paired seed-0 traces. |
| M2, initial contact | A scripted position command physically pushed a 1 kg box in MuJoCo and Isaac. Contact timing, peak force and lateral slip differ as quantified above. Evaluate prefix prediction error on held-out approaches before using contact forces in an MPC cost. |
| M3, parallel rollouts | Snapshot restoration, candidate isolation, serial/parallel equivalence and a timed batch during box contact passed. |
| M4, box MVP | Passed the near-box, position-only goal task and matched hold/scripted comparison. Executed five-tick prefixes matched their forecasts exactly, including during contact. |
| M5, cone and controls | Position-only MPC uprighted and released the fallen cone. Ten paired box starts/masses compared position-only, force-enabled and matched-effective-position commands at equal budgets. |
| M6 | Use the same scene, fidelity, seeds, number of scored candidates/mean passes, horizon, prefix and policy/physics ticks on CPU MuJoCo, GPU PhysX and optimized CPU PhysX. Time warm end-to-end planning, snapshot, schedule generation, inference, actuators, physics/contact, cost/selection and live execution. Report per-rollout work and wall time, p50/p95/max, fraction exceeding the 100 ms prefix, and closed-loop success. Start with eight candidates, three knots, one iteration and one ORT thread. Compare serial, a small persistent pool and ORT thread settings in short matched runs. |

In a longer seed-0 check, forward-speed MPC reached a 50 cm goal with 9.8 mm final error in 4.9 control seconds. Base advance was 55.2 cm and the furthest issued finger target was X = 1.270 m against the fixed 1.280 m upper bound. This trial succeeded with 10 mm of remaining command range. [50 cm run](artifacts/mobile_box/mobile_50cm/summary.json)

**Cone objective and success.** The implemented objective averages cone-axis alignment error and finger approach over the horizon, then encourages release when the axis reaches SUMO’s upright threshold. The separate success gate requires 0.5 s within 10° of world vertical with low angular speed, ground support and no active hand contact. [Cone experiments](docs/cone_uprighting.md)

The opt-in force-conditioned box planner and ordinary position MPC each precision-settled seven of ten mass/start conditions. The matched-offset position planner settled six. Measured wall-normal load and virtual force command differed substantially. [Force interface, comparisons and measured loads](docs/ee_force_box.md)

### Synchronized episode dashboards

Add `--dashboard` to the offline renderer for an annotated side/overhead MP4 and a static PNG. New episode recordings include each winning full-horizon forecast. The plots align raw requests, projected and effective targets, actor inputs, measured contact, body velocities and task outcomes with the executed history. Rendering takes place after the controller exits. [Launch command, timing, units and sign conventions](docs/episode_telemetry.md) · [Paired optimizer experiments](docs/exploration.md)
