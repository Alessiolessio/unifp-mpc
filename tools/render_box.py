"""Render an executed box episode at simulation speed. Requires an FFmpeg build with subtitles."""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import mujoco
import numpy as np


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
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    summary = json.loads((args.episode / "summary.json").read_text())
    model = mujoco.MjModel.from_binary_path(summary["scene"])
    model.geom("ground").group = 0
    data = mujoco.MjData(model)
    goal = np.array(summary["goal_xy_m"])
    side, top = mujoco.MjvCamera(), mujoco.MjvCamera()
    side.lookat[:] = [.43, 0, .43]
    side.azimuth, side.elevation, side.distance = 100, -15, 2.55
    top.lookat[:] = [*np.mean([summary["box_start_xy_m"], goal], axis=0), .50]
    top.azimuth, top.elevation, top.distance = 90, -90, .74
    video = (args.episode / "episode.mp4").resolve()
    with np.load(args.episode / "episode.npz") as trace, tempfile.TemporaryDirectory() as tmp:
        frames = np.arange(4, len(trace["time_s"]), 5)
        captions = []
        for n, i in enumerate(frames):
            error = np.linalg.norm(trace["qpos"][i, -7:-5] - goal)
            mode = "Warmup" if i < 50 else summary["controller"].upper()
            captions.append(f"{n+1}\n{stamp(n/10)} --> {stamp((n+1)/10)}\n"
                            f"{mode}   {trace['time_s'][i]:.1f} s     |     "
                            f"Box center (white)    Goal (green)     |     XY error  {100*error:.1f} cm\n")
        Path(tmp, "labels.srt").write_text("\n".join(captions))
        args_ff = [args.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
                   "-pix_fmt", "rgb24", "-s", "960x400", "-r", "10", "-i", "-", "-vf",
                   "subtitles=labels.srt:force_style='Fontname=DejaVu Sans,Fontsize=14,MarginV=5'",
                   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)]
        with mujoco.Renderer(model, 360, 640) as renderer, subprocess.Popen(
            args_ff, cwd=tmp, stdin=subprocess.PIPE
        ) as writer:
            frame = np.full((400, 960, 3), 22, dtype=np.uint8)
            for i in frames:
                data.qpos[:] = trace["qpos"][i]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=side)
                frame[:360, :640] = renderer.render()
                renderer.update_scene(data, camera=top)
                mark_centers(renderer.scene, data.qpos[-7:-4], goal, summary["tolerance_m"])
                frame[:360, 640:] = renderer.render()[:, 160:480]
                writer.stdin.write(frame.tobytes())
            writer.stdin.close()
            if writer.wait():
                raise RuntimeError("FFmpeg could not encode the episode")
    print(video)


if __name__ == "__main__":
    main()
