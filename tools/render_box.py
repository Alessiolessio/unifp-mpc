"""Render an executed object manipulation episode at simulation speed. Requires an FFmpeg build with subtitles."""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import mujoco
import numpy as np


def draw_path(scene, points, color, *, dashed=False):
    for i, (a, b) in enumerate(zip(points[:-1], points[1:])):
        if (dashed and i % 2) or np.array_equal(a, b):
            continue
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), a,
                            np.eye(3).ravel(), color)
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 3, a, b)
        scene.ngeom += 1


def mark_centers(scene, box, goal, tolerance):
    # The two XY markers sit just above the box lid in the overhead view.
    height = box[2] + .26
    circle = np.linspace(0, 2*np.pi, 33)
    ring = np.column_stack((goal[0] + tolerance*np.cos(circle), goal[1] + tolerance*np.sin(circle),
                            np.full(len(circle), height)))
    for a, b in zip(ring[:-1], ring[1:]):
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), a,
                            np.eye(3).ravel(), [0.1, 1., .35, 1.])
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 3, a, b)
        scene.ngeom += 1
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, [.006]*3,
                        [box[0], box[1], height+.001], np.eye(3).ravel(), [1., 1., 1., 1.])
    scene.ngeom += 1


def stamp(t):
    ms = round(t * 1000)
    return f"{ms//3600000:02}:{ms//60000%60:02}:{ms//1000%60:02},{ms%1000:03}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path, help="Directory containing episode.npz and summary.json")
    parser.add_argument("--dashboard", action="store_true",
                        help="Write dashboard.mp4 and dashboard.png with synchronized telemetry (Matplotlib)")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    summary = json.loads((args.episode / "summary.json").read_text())
    model = mujoco.MjModel.from_binary_path(summary["scene"])
    model.geom("ground").group = 0
    data = mujoco.MjData(model)
    cone = summary.get("task") == "cone"
    wall = summary.get("task") == "wall"
    tether = summary.get("task") == "tether"
    loaded_walk = summary.get("task") == "base_load"
    if cone:
        from kinematics import angle_deg, cone_axis, show_cone_mesh
        show_cone_mesh(model)
    else:
        goal = np.array(summary["goal_xy_m"])
    side, top = mujoco.MjvCamera(), mujoco.MjvCamera()
    side.lookat[:] = [.43, 0, .43]
    side.azimuth, side.elevation, side.distance = 100, -15, 2.55
    if wall:
        side.lookat[:] = [.47, goal[1]/2, .48]
        side.azimuth, side.elevation = 55, -19
        top.lookat[:] = [.40, goal[1]/2, .35]
    elif loaded_walk:
        side.lookat[:] = [*(goal/2), .4]
        side.distance = 3.0
        top.lookat[:] = [*(goal/2), .1]
    elif not cone:
        top.lookat[:] = [*np.mean([summary["box_start_xy_m"], goal], axis=0), .50]
    top.azimuth, top.elevation, top.distance = 90, -90, .74
    video = (args.episode / ("dashboard.mp4" if args.dashboard else "episode.mp4")).resolve()
    with np.load(args.episode / "episode.npz") as trace, tempfile.TemporaryDirectory() as tmp:
        # The overhead panel crops the central 320 pixels of a 360-pixel-high camera.
        poses = trace["qpos"]
        centers = poses[:, -7:-5] if cone else np.vstack((poses[:, -7:-5], goal))
        if wall:
            centers = np.vstack((poses[:, :2], trace["ee_world"][:, :2], goal))
        elif loaded_walk:
            centers = np.vstack((poses[:, :2], goal))
        if cone:
            centers = np.vstack((centers, centers + .673 * cone_axis(poses[:, -4:])[:, :2]))
            top.lookat[:] = [*((centers.max(0) + centers.min(0)) / 2), .25]
        extent = np.max(np.abs(centers - top.lookat[:2])) + np.hypot(.2, .2)
        half_fov = np.deg2rad(model.vis.global_.fovy / 2)
        top.distance = max(top.distance, .03 + extent / (np.tan(half_fov) * 320 / 360))
        if tether:
            xy = np.vstack((poses[:, :2], centers))
            side.lookat[:] = [*((xy.min(0)+xy.max(0))/2), .32]
            side.distance = 2.6 + .65*np.linalg.norm(np.ptp(xy, axis=0))
        if "scenario" in summary:
            xy = np.vstack((trace["qpos"][:, :2], centers))
            side.lookat[:2] = (xy.min(axis=0) + xy.max(axis=0)) / 2
            side.distance = 2.2 + np.linalg.norm(np.ptp(xy, axis=0))
            side.azimuth += np.rad2deg(summary["scenario"]["robot_yaw_rad"])
        stride = 2 if cone or wall or loaded_walk or tether else 5
        fps = 50 // stride
        frames = np.arange(stride-1, len(poses), stride)
        projected = trace["projected"] if "projected" in trace.files else None
        times = trace["time_s"] if "time_s" in trace else (np.arange(len(poses))+1)*.02
        dashboard = None
        if args.dashboard:
            from episode_plots import Dashboard
            dashboard = Dashboard(args.episode, trace, summary, model.key("spawn").qpos[:7])
            if frames.size == 0 or frames[-1] != len(poses)-1:
                frames = np.append(frames, len(poses)-1)
        captions = []
        tensions = trace["tension_N"].mean(axis=1) if tether else None
        if loaded_walk:
            physical = trace["load_world_N"]
            virtual_force = np.linalg.norm(trace["policy_command"][:, 9:11], axis=1)
        for n, i in enumerate(frames):
            mode = "Wall MPC" if wall else (
                "Warmup" if i < 50 else "MPC" if cone else summary["controller"].upper())
            if cone:
                detail = f"Tilt {angle_deg(poses[i, -4:]):.1f} deg   |   Cyan request, amber finger"
                if projected is not None:
                    detail += ", violet issued position"
            elif wall:
                load = trace["gripper_on_wall_N"][i, 0]
                detail = (f"{summary['ee_force']}   |   Measured {load:.1f} N   "
                          f"Setpoint {summary['desired_N']:g} N   |   "
                          f"Base goal error {100*np.linalg.norm(poses[i,:2]-goal):.1f} cm")
            elif tether:
                mode = "Tether MPC | " + summary["controller"]
                tension = tensions[i]
                error = np.linalg.norm(poses[i, -7:-5] - goal)
                detail = f"Cable {tension:.1f} N   |   Box goal error {100*error:.1f} cm"
                if n == len(frames)-1:
                    detail += "   |   " + summary["status"].replace("_", " ")
            elif loaded_walk:
                mode = "Base MPC | " + summary["controller"]
                fx, fy = physical[i]
                virtual = virtual_force[i]
                detail = (f"World load ({fx:+.0f}, {fy:+.0f}) N   |   "
                          f"Virtual command {virtual:.1f} N   |   "
                          f"Goal error {100*np.linalg.norm(poses[i,:2]-goal):.1f} cm")
            else:
                error = np.linalg.norm(poses[i, -7:-5] - goal)
                detail = f"Box center (white)    Goal (green)     |     XY error  {100*error:.1f} cm"
            captions.append(f"{n+1}\n{stamp(n/fps)} --> {stamp((n+1)/fps)}\n"
                            f"{mode}   {times[i]:.1f} s     |     {detail}\n")
        Path(tmp, "labels.srt").write_text("\n".join(captions))
        size = "1600x1000" if dashboard else "960x400"
        args_ff = [args.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
                   "-pix_fmt", "rgb24", "-s", size, "-r", str(fps), "-i", "-"]
        if not dashboard:
            args_ff += ["-vf", "subtitles=labels.srt:force_style='Fontname=DejaVu Sans,Fontsize=14,MarginV=5'"]
        args_ff += ["-c:v", "libx264", "-threads", "2", "-pix_fmt", "yuv420p", "-movflags",
                    "+faststart", str(video)]
        targets = trace["targets"] if cone and "targets" in trace else None
        fingers = trace["fingers"] if targets is not None else None
        wall_paths = trace["paths"] if wall else None
        executed_finger = trace["ee_world"] if cone else None
        with mujoco.Renderer(model, 360, 640) as renderer, subprocess.Popen(
            args_ff, cwd=tmp, stdin=subprocess.PIPE
        ) as writer:
            frame = np.full((400, 960, 3), 22, dtype=np.uint8)
            for i in frames:
                data.qpos[:] = poses[i]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=side)
                if cone:
                    origin = poses[i, -7:-4]
                    draw_path(renderer.scene, [origin, origin + [0., 0., .76]], [.1, 1., .35, 1.])
                    if targets is not None and len(targets) and i+1 >= 50:
                        update = min((i+1-50)//5, len(targets)-1)
                        draw_path(renderer.scene, targets[update], [0., .9, 1., 1.])
                        if projected is not None:
                            draw_path(renderer.scene, projected[update], [.82, .35, 1., 1.], dashed=True)
                        start = executed_finger[49 + 5*update]
                        draw_path(renderer.scene, np.vstack((start, fingers[update])), [1., .57, .03, 1.])
                        mujoco.mjv_initGeom(renderer.scene.geoms[renderer.scene.ngeom],
                                            mujoco.mjtGeom.mjGEOM_SPHERE, [.012]*3, start,
                                            np.eye(3).ravel(), [1., 1., 1., 1.])
                        renderer.scene.ngeom += 1
                if loaded_walk:
                    start = data.xipos[model.body("body").id]
                    force = physical[i]
                    draw_path(renderer.scene, [start, start + np.r_[force*.015, .0]],
                              [1., .35, .15, 1.])
                if wall:
                    raw, projected_wall, predicted = wall_paths[min((i+1)//5, len(wall_paths)-1)]
                    draw_path(renderer.scene, raw, [0., .9, 1., 1.])
                    draw_path(renderer.scene, projected_wall, [.82, .35, 1., 1.], dashed=True)
                    draw_path(renderer.scene, predicted, [1., .57, .03, 1.])
                frame[:360, :640] = renderer.render()
                renderer.update_scene(data, camera=top)
                if wall or loaded_walk:
                    ring = np.linspace(0, 2*np.pi, 33)
                    radius = .05 if loaded_walk else .075
                    draw_path(renderer.scene, np.column_stack((
                        goal[0]+radius*np.cos(ring), goal[1]+radius*np.sin(ring),
                        np.full(len(ring), .01))), [.1, 1., .35, 1.])
                elif not cone:
                    mark_centers(renderer.scene, data.qpos[-7:-4], goal, summary["tolerance_m"])
                frame[:360, 640:] = renderer.render()[:, 160:480]
                picture = dashboard.frame(i, frame) if dashboard else frame
                writer.stdin.write(picture.tobytes())
                if dashboard and i == frames[-1]:
                    from matplotlib.image import imsave
                    imsave(video.with_suffix(".png"), picture)
            writer.stdin.close()
            if writer.wait():
                raise RuntimeError("FFmpeg could not encode the episode")
    print(video)


if __name__ == "__main__":
    main()
