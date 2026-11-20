# Live randomized box MPC

`pixi run --locked view-box` runs the mobile controller with 24 candidates, one CEM iteration, three knots, a 0.5 s horizon and a 0.1 s execution prefix. The default CEM seed is 17. Each episode increments the scenario seed and cycles easy → medium → hard. The log records both seeds, tier and planner settings for individual replay. The control episode ends after 9 simulated seconds, a failed plan or live trajectory failure, or after the box spends 25 consecutive policy ticks within 15 mm of the goal with XY speed at most 2 cm/s over the last five ticks. Each episode starts with a one-second contact-free warmup.

The passive MuJoCo window has a separate visualization model. The display thread consumes the previous executed prefix while native workers evaluate the next one. Cyan shows the sampled world targets before the policy's spherical workspace projection (time 0 through the request held over the final tick). Amber starts at the live finger and continues through the selected rollout's post-action finger positions, ending at the horizon. The amber dot tracks the executed finger. The white dot marks the box center in XY at lid height, and the green ring marks the final goal at the same height. Camera controls affect presentation only. Each episode starts with a camera view across the pushing direction. MuJoCo supports syncing integration state and user-drawn geoms together. [Viewer](../tools/view_box.py) · [MuJoCo 3.6 sync implementation](https://github.com/google-deepmind/mujoco/blob/3.6.0/simulate/simulate.cc#L2137-L2169)

SUMO inherits Judo's visualization node, where live poses and forecast traces arrive separately. Here both come directly from the existing controller. The viewer reads the winning batch member before the CEM arrays are released. The render and simulation models have the same geometry. On reset the generator writes one temporary episode model with the new pose and friction, and the native pool binds the fresh live world and worker data to that immutable model. The ONNX session and rollout threads live for the duration of the process. CEM, history, gait clock, previous actions, delay queues and contact buffers start fresh. [Judo visualization node](../../judo/judo/app/dora/visualization_node.py) · [Native episode reset](../native/rollouts.h)

## Placement and reachability

| Tier | Forward center distance | Lateral offset | Box yaw | Goal direction | Push distance | Floor / hand sliding friction |
|---|---:|---:|---:|---:|---:|---|
| Easy | 1.04–1.10 m | ±5 cm | ±15° | ±12° | 8–22 cm | .24–.36 / .27–.43 |
| Medium | 1.12–1.34 m | ±12 cm | ±35° | ±25° | 16–38 cm | .20–.44 / .22–.50 |
| Hard | 1.38–1.65 m | ±20 cm | ±45° | ±40° | 24–52 cm | .18–.48 / .20–.54 |

Angles and offsets in the table use the starting robot heading. Robot XY varies by ±0.4 m per axis and world yaw spans ±180°. The goal is an offset from the actual settled box center. A MuJoCo contact check rejects overlapping spawns and the executed warmup checks for gripper or other robot contact. Live and forecast physics match. Judo's box reset samples a full circle and adds position noise. The forward sector here gives the body-forward-speed planner meaningful approach tasks. Tiered scenes use a rectangle aligned with initial robot yaw. It includes the warmup request, then tracks the box while retaining the last issued request in bounds. Commands sent to the live controller and rollouts remain world XYZ plus body-forward speed. Historical scenes keep their original workspace behavior. [Generator](../tools/box_scenario.py) · [Judo reset](../../judo/judo/tasks/spot/spot_box_push.py)

## Paired cost and horizon checks

Final XY error in mm, CEM seed 17, 9 s control limit. The three costs share the bounded workspace guidance and equal running/terminal local box error. Only the finger approach distance varies. **Directional** uses the rear pushing face. **Proximity** uses squared 3D finger-to-box-center distance with the same weight. **Hybrid** uses proximity beyond a 55 cm live finger/center gap, directional inside 35 cm, and interpolates their squared costs between. SUMO's object-speed penalty was left out of this ablation. [Cost implementation](../tools/box_mpc.py) · [SUMO box reward](../../sumo/sumo/tasks/spot/spot_box_push.py)

| Scene / seed | Directional | Proximity | Hybrid |
|---|---:|---:|---:|
| Easy / 0 | 0.2 | 1.6 | 3.8 |
| Medium / 1 | 7.2 | 0.8 | 5.1 |
| Hard / 2 | 3.3 | 2.0 | 4.4 |
| Hard / 7 | 8.3 | 4.7 | 13.6 |
| Hard / 9 | 19.3 (timeout) | 105.6 (timeout) | 35.4 (timeout) |
| Settled / attempted | 4/5 | 4/5 | 4/5 |

On hard seed 2, all 24 first-update candidates predicted identical box XY and none reached contact. The directional and proximity costs still spanned 0.000215 and 0.000118 m² across the batch. The executed finger first contacted the box at 2.18 s with directional guidance and 1.78 s with proximity or hybrid. On hard seed 9, proximity reached contact at the same 2.18 s as directional guidance, but had 34 contact ticks against 68 and a 110 mm maximum cross-track error against 68 mm. All five scenes stayed within 8.2° roll/pitch and had zero nongripper impulse for all three costs. Directional guidance remains the default. The five starts and one optimizer seed do not establish a population success rate.

| Hard seed 2, directional, 24 candidates | First contact after warmup | Control time to settle | Full planning p95 | Misses / updates |
|---|---:|---:|---:|---:|
| 0.5 s horizon | 2.18 s | 4.3 s | 49.2 ms | 0/43 |
| 1.0 s | 2.08 s | 4.9 s | 100.0 ms | 3/49 |
| 1.5 s | 2.22 s | 6.6 s | 140.8 ms | 66/66 |

## Display cost and checks

The paced window and headless loop ran consecutive easy/medium/hard episodes using seeds 0/1/2, eight rollout workers and the default planner. Planning includes sampling, native forecasts, cost, selection and the optional winning-path copy. Wall time below includes episode setup and warmup and excludes startup and the between-episode pause.

| Mode | Episode planning p50 range | Episode planning p95 range | Maximum | Deadline misses | Total simulated / wall time |
|---|---:|---:|---:|---:|---:|
| Headless | 45.0–47.7 ms | 51.4–53.9 ms | 56.4 ms | 0/97 | 12.7 / 5.11 s |
| Interactive, paced | 59.0–60.1 ms | 67.1–71.7 ms | 79.9 ms | 0/97 | 12.7 / 13.48 s |

The window presented 369 distinct executed poses in 12.95 s of active playback (28.5 pose frames/s, 0.98× real time). This counts state presentations to the window. Monitor refresh and repeated frames from MuJoCo's camera loop are separate. All three episodes matched headless goal errors, control ticks and gripper contact counts. The live-versus-forecast maximum prefix error was zero throughout. A real-window smoke check also exercised timeout, reset and closing during active playback. Fresh versus recycled worlds matched exactly after warmup, prior model snapshots were rejected, and all 25/50/75 plotted finger positions matched independent replays of the selected commands. A regenerated legacy seed 12 reproduced the archived non-timing trace and plan arrays exactly. [Reset and path check](../tests/check_view_box.py) · [Window close check](../tests/check_window_close.py)

To reproduce a cost comparison, run `pixi run --locked view-box --headless --difficulty hard --scenario-seed 9 --cem-seed 17 --episodes 1 --objective proximity`. Use `--horizon 1` or `--horizon 1.5` to repeat the longer-horizon tests on seed 2. The next targeted comparison is lateral contact recovery from the hard seed 9 trajectory, using the existing optional lateral command before expanding the optimizer to yaw.
