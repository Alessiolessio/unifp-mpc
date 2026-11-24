# Reading an MPC episode dashboard

From the repository root, render a recorded episode with MuJoCo 3.6.0, Matplotlib 3.10.8 and an FFmpeg encoder with libx264. The offline renderer needs neither the native policy module nor a running planner. For example, with uv and FFmpeg on the path:

```sh
MUJOCO_GL=egl uv run --no-project --python 3.12 --with mujoco==3.6.0 --with matplotlib==3.10.8 \
  tools/render_box.py artifacts/telemetry/cone-67 --dashboard
```

This writes `dashboard.mp4` and a full-history `dashboard.png` beside the recording. Supply `--ffmpeg /path/to/ffmpeg` for a non-system encoder. Omitting `--dashboard` uses the original camera/subtitle layout. On hosts with the existing simulation environment, Matplotlib is only needed for the dashboard. [Renderer](../tools/render_box.py) · [Plotted signals](../tools/episode_plots.py)

## Reading time and forecasts

Each tick has a pre-action root and command at time *t*, four 5 ms physics steps, and a state, finger position and interval contact measurement at *t* + 20 ms. The solid lines show executed history. Dashed lines show the selected candidate across its full horizon. The vertical cursor and both camera panes show the same post-action instant. At a five-tick boundary the next winning horizon starts at that instant. At termination the last selected forecast remains visible. Its tail was hypothetical. A frame can span more than one policy step. The time in the heading is the simulation clock for the saved trace. Box and cone include the 1 s warmup. Wall, walking and towing start their saved clocks after warmup.

New recordings store `forecast_start_tick` as an index **between** ticks in `episode.npz`. Entry 50 follows executed row 49. For each replan the recorder saves the pre-action root, the winner's requested schedule, issued policy commands, predicted states and task measurements. The first five predicted post-action rows align with the five executed rows beginning at that index. Compression and plotting run outside planning. A rejected batch can have a latency and no winning forecast. Old files display only the fields they recorded. Legacy cone/wall path arrays supply actual historical forecasts where available. The renderer leaves missing quantities out. [Recorder](../tools/forecasts.py) · [Prefix and archival parity check](../tests/check_forecast_recording.py)

## Quantities and signs

| Panel | Definition |
|---|---|
| EE world position, m | **Request** is the raw CEM world XYZ. **Nominal** is the world target after projection into the pre-action spherical workspace, before any force offset. **Effective** decodes the spherical command actually issued and adds the issued virtual EE force rotated into world axes divided by 200 N/m. **Finger** is the post-action physical finger-body origin. In matched-position mode the actor receives zero virtual force and the issued spherical command already encodes the reachable offset. The nominal projection of the original request is reconstructed using the same radius, elevation and azimuth bounds. |
| Live violet EE path | This decodes the issued spherical command. It shows nominal position in conditioned mode and position with the reachable offset in matched mode. The offline dashboard shows both quantities separately. |
| EE world force, N | **Virtual proposal** is the optimizer's proposed action or offset parameter. Eight-column horizontal force requests are already world-frame. Nine-column cone force requests rotate from the pre-action yaw frame. **Actor input** rotates issued policy slots 6–8 by the same yaw, after shortening. In the matched mode the proposal parametrizes an offset and actor force input is zero. **Hand on object** sums the world-frame gripper forces applied to the task object and averages over four physics steps. The object-on-hand reaction has the opposite sign. Virtual force is an actor input. |
| Base velocity | Commanded translation and rotation use UniFP policy slots 0–2. Measured translation uses the post-action inverse **full** base quaternion on world free-joint linear velocity, taking body X/Y in m/s as in the tracking reward. Angular velocity is MuJoCo free-joint body Z in rad/s, as tracked by the policy. This component can differ from the time derivative of Euler yaw. |
| Loaded-walk velocity target | In yaw axes the target XY is ordinary command + (virtual base force + applied external torso force)/200. It is zeroed when both adjusted components are within 0.1 m/s and the yaw command is within 0.2 rad/s. The commanded torso load is recorded separately in world axes. Unannounced pulse edges are absent from forecasts made before those edges. |
| Base world force, N | Virtual actor slots 9–11 rotate from the pre-action yaw frame. Loaded walking displays the external torso input actually applied through MuJoCo. Towing reconstructs the load on the torso as **minus** tick-mean scalar tension times the unit eye-to-hitch direction. The saved cable direction is sampled at tick end, so this vector is an approximation to the substep-averaged load. There is no inferred ground-reaction equivalent of the actor command. |
| Task panels | Box XY and distance to the fixed XY goal; cone axis tilt, norm of object angular velocity, ground vertical support and hand contact duty; desired and tick-average measured wall normal; box-goal error and scalar tendon tension during towing; distance to the time-varying reference during loaded walking. All plotted distances are metres. |

Forecasts and measurements use separate native buffers. On the six reference recordings, both cone outcomes, the matched wall run, the conditioned box push, the diagonal tow and the unexpected-load walk reproduced their archived policy commands and physical trajectories exactly. The winning five-tick prefixes matched every checked modeled tick. On the unexpected-load walk, six of 250 ticks followed an unannounced load change within the current prefix; the maximum forecast error in base XY was 1.26 mm. [Signal decoding](../tools/kinematics.py) · [Native record layout](../native/world.cpp)

## Rendered examples

| Episode | Video | Full-history plot |
|---|---|---|
| Strict cone release, 82/67 | [MP4](../artifacts/telemetry/cone-67/dashboard.mp4) | [PNG](../artifacts/telemetry/cone-67/dashboard.png) |
| Stalled cone, 82/43 | [MP4](../artifacts/telemetry/cone-43/dashboard.mp4) | [PNG](../artifacts/telemetry/cone-43/dashboard.png) |
| Ordinary cone release with four broad proposals, 419/137 | [MP4](../artifacts/exploration/419/137-off-broad/dashboard.mp4) | [PNG](../artifacts/exploration/419/137-off-broad/dashboard.png) |
| Conditioned box push | [MP4](../artifacts/telemetry/box/dashboard.mp4) | [PNG](../artifacts/telemetry/box/dashboard.png) |
| Diagonal tethered pull | [MP4](../artifacts/telemetry/tether/dashboard.mp4) | [PNG](../artifacts/telemetry/tether/dashboard.png) |
| Matched-position wall contact | [MP4](../artifacts/telemetry/wall/dashboard.mp4) | [PNG](../artifacts/telemetry/wall/dashboard.png) |
| Unexpected torso-load pulse | [MP4](../artifacts/telemetry/walk/dashboard.mp4) | [PNG](../artifacts/telemetry/walk/dashboard.png) |
