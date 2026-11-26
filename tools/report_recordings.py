"""Prepare selected camera replays in temporary folders; never write beside archived runs."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

import mujoco
import numpy as np

DEST = Path("docs/report_media")


def render(folder, trace, summary, name, ffmpeg):
    (folder/"summary.json").write_text(json.dumps(summary))
    with (folder/"episode.npz").open("wb") as f:
        np.savez(f,**trace)
    subprocess.run([sys.executable,"tools/render_box.py",str(folder),"--ffmpeg",ffmpeg],check=True)
    shutil.copyfile(folder/"episode.mp4", DEST/(name+".mp4"))


def camera_strip(folder, poses, models, labels, name, ffmpeg, goal=None):
    """Synchronized physical views of archived poses. Captions use post-action time."""
    from contextlib import ExitStack
    from render_box import mark_centers, stamp
    total = max(map(len,poses))
    indices = list(range(1,total,2))
    subtitles = []
    for i,t in enumerate(indices):
        caption = "      |      ".join(f"{label} (ended)" if t>=len(q) else label
                                      for q,label in zip(poses,labels))
        subtitles.append(f"{i+1}\n{stamp(i/25)} --> {stamp((i+1)/25)}\n"
                         f"{(t+1)*.02:.2f} s  |  {caption}\n")
    (folder/"labels.srt").write_text("\n".join(subtitles))
    target = (DEST/(name+".mp4")).resolve()
    cmd = [ffmpeg,"-v","error","-y","-f","rawvideo","-pix_fmt","rgb24","-s",
           f"{384*len(poses)}x288","-r","25","-i","-","-vf",
           "subtitles=labels.srt:force_style='Fontname=DejaVu Sans,Fontsize=13,MarginV=5'",
           "-c:v","libx264","-threads","2","-pix_fmt","yuv420p","-movflags","+faststart",str(target)]
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [.44,0,.42]
    camera.azimuth,camera.elevation,camera.distance = 98,-16,2.6
    if name.startswith("wall"):
        camera.azimuth, camera.elevation = 55, -19
    if name.startswith("tow"):
        camera.lookat[0]=-.20
        camera.distance=3.6
    if name.startswith("relic") or goal is not None:
        xy = np.concatenate([q[:,0:2] for q in poses]+[q[:,-7:-5] for q in poses])
        camera.lookat[:2] = (xy.min(0)+xy.max(0))/2
        view = poses[0][0, -7:-5] - poses[0][0, :2]
        camera.azimuth = np.rad2deg(np.arctan2(view[1], view[0])) + 90
        camera.elevation = -22
        camera.distance = 2.6 + .8*np.linalg.norm(np.ptp(xy, axis=0))
    for model in models:
        model.geom("ground").group = 0
    with ExitStack() as stack:
        renderers = [stack.enter_context(mujoco.Renderer(m,288,384)) for m in models]
        data = [mujoco.MjData(m) for m in models]
        w = stack.enter_context(subprocess.Popen(cmd,stdin=subprocess.PIPE,cwd=folder))
        for t in indices:
            parts=[]
            for r,m,d,q in zip(renderers,models,data,poses,strict=True):
                d.qpos[:]=q[min(t,len(q)-1)]
                mujoco.mj_forward(m,d)
                r.update_scene(d,camera=camera)
                if goal is not None:
                    mark_centers(r.scene, d.qpos[-7:-4], goal, .015)
                parts.append(r.render().copy())
            w.stdin.write(np.concatenate(parts,axis=1).tobytes())
        w.stdin.close()
        if w.wait():
            raise RuntimeError("Camera encoding failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family",choices=["box","slide","relic","pulse","raw","wall","mismatch","relic_cone"])
    parser.add_argument("--ffmpeg",default="ffmpeg")
    args = parser.parse_args()
    DEST.mkdir(exist_ok=True)
    with TemporaryDirectory() as tmp:
        folder = Path(tmp)
        if args.family=="pulse":
            from fixed_finger_pulse import run
            rows = run(4201,mujoco.MjModel.from_binary_path("artifacts/spot.mjb"),
                       folder,synchronized=True)
            with open("docs/final_pulse.csv") as f:
                old = [r for r in csv.DictReader(f) if r["case"]=="4201"]
            for r,s in zip(rows,old,strict=True):
                assert r["mode"]==s["mode"]
                for k in r.keys()-{"mode"}:
                    np.testing.assert_allclose(r[k],float(s[k]),rtol=1e-12,atol=1e-12)
                name = f"pulse_4201_{r['mode']}.npz"
                shutil.copyfile(folder/name, DEST/name)
        elif args.family=="mismatch":
            root = Path("artifacts/supplemental/uncertainty/box/5503")
            source = root / "torso/nominal_prediction"
            model = mujoco.MjModel.from_binary_path(str(root / "torso.mjb"))
            summary = json.loads((source / "A/summary.json").read_text())
            poses = []
            for arm in "ABC":
                with np.load(source / arm / "episode.npz") as trace:
                    poses.append(trace["qpos"][50:])
            camera_strip(folder, poses, [model]*3, ["A ordinary", "B force", "C matched"],
                         "torso_5503_451", args.ffmpeg, np.array(summary["goal_xy_m"]))
        elif args.family=="wall":
            from prepare_scene import scene_template
            with open("docs/supplement_wall.csv") as f:
                row = next(r for r in csv.DictReader(f) if r["case"] == "5201")
            model = scene_template(wall_face_x=float(row["face_x_m"]))
            poses = []
            for arm in "ABC":
                with np.load(f"artifacts/supplemental/wall/5201/{arm}.npz") as trace:
                    poses.append(trace["qpos"])
            camera_strip(folder, poses, [model]*3, ["A ordinary", "B +10 N", "C matched"],
                         "wall_5201", args.ffmpeg)
        elif args.family=="raw":
            from fixed_policy_loads import prepare_case
            for task,name,level in [("standing_payload","stand",2),("walking_payload","walk",2),
                                    ("tether","tow",-16)]:
                model,*_ = prepare_case(task,100,False,folder/"fixture.mjb",None)
                poses = []
                for arm in "ABC":
                    suffix = f"{arm}{0 if arm=='A' else level:+g}"
                    with np.load(f"artifacts/fixed_policy_loads/held/{task}_100_{suffix}.npz") as t:
                        poses.append(t["pose"])
                camera_strip(folder,poses,[model]*3,["A ordinary","B conditioned","C matched"],
                             name+"_100",args.ffmpeg)
        elif args.family=="relic_cone":
            import judo.tasks.base as base
            from sumo.tasks.spot.spot_cone_upright import SpotConeUpright
            from kinematics import show_cone_mesh
            (folder/"meshes").symlink_to(base.MODEL_PATH/"meshes",target_is_directory=True)
            base.MODEL_PATH = folder
            src = Path("artifacts/final/3109")
            models = [mujoco.MjModel.from_binary_path(str(src/"scene.mjb")),SpotConeUpright().model]
            show_cone_mesh(models[0])
            paths = [src/"311-known-A", Path("artifacts/supplemental/sumo/paired/3109/311")]
            poses = []
            for path,start in zip(paths,[50,0]):
                with np.load(path/"episode.npz") as tr:
                    poses.append(tr["qpos"][start:])
            camera_strip(folder,poses,models,["UniFP ordinary","ReLIC runtime"],
                         "relic_cone_3109_311",args.ffmpeg)
        elif args.family=="box":
            base = Path("artifacts/supplemental/push/replay/4101")
            with np.load(base/"warm.npz") as warm:
                warm_poses = warm["qpos"]
            for arm,desc in zip("ABC",["ordinary","force +12 N","matched +60 mm"],strict=True):
                with np.load(base/(arm+".npz")) as t:
                    goal = t["goal_xy"]
                    s = dict(scene="artifacts/spot_box.mjb",goal_xy_m=goal.tolist(),
                             box_start_xy_m=(goal-[.05,0]).tolist(),tolerance_m=.015,
                             controller=f"Fixed 4101 {arm} {desc}")
                    poses = np.vstack((warm_poses, t["qpos"]))
                    time = .02 * (np.arange(1, len(poses)+1) - len(warm_poses))
                    render(folder, {"qpos":poses, "time_s":time}, s, "push_4101_"+arm, args.ffmpeg)
        elif args.family=="slide":
            path = Path("artifacts/uncertainty/wall/9/conditioned")
            s = json.loads((path/"summary.json").read_text())
            s["scene"] = "artifacts/wall_layer_tick/9/scene.mjb"
            with np.load(path/"episode.npz") as trace:
                render(folder,dict(trace),s,"sliding_91_67_B",args.ffmpeg)
        else:
            from sumo_common import SpotBoxPush, base, shared_box, verify
            (folder/"meshes").symlink_to(base.MODEL_PATH/"meshes",target_is_directory=True)
            base.MODEL_PATH = folder
            src = mujoco.MjModel.from_binary_path("artifacts/final/3015/scene.mjb")
            class Fixture(SpotBoxPush):
                def _process_spec(self):
                    super()._process_spec()
                    shared_box(self,src)
            task = Fixture()
            model = task.model
            verify(src,model,"box_body")
            base_path = Path("artifacts/final/3015")
            summary = json.loads((base_path/"311-known-A/summary.json").read_text())
            with np.load(base_path/"311-known-native/episode.npz") as relic:
                with np.load(base_path/"311-known-A/episode.npz") as ordinary:
                    camera_strip(folder, [ordinary["qpos"][50:], relic["qpos"]], [src, model],
                                 ["UniFP ordinary", "ReLIC native"], "relic_box_3015_311",
                                 args.ffmpeg, np.array(summary["goal_xy_m"]))


if __name__=="__main__":
    main()
