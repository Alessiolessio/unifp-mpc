"""Fixed high-level contact with the existing rigid wall, including unloaded intervals."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np

from fixed_policy_loads import DT, N, command, ee_goal, length, rms
from final_policy import save
from physical_score import tilt
from prepare_scene import scene_template
from run_world import command_schedule, spherical
from unifp_native import World

ROOT = Path("artifacts/supplemental/wall")


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    wall = scene_template(wall_face_x=.89)
    rows = []
    with TemporaryDirectory(dir=ROOT) as tmp:
        scene = str(Path(tmp)/"wall.mjb")
        for case in [0,1,*range(5201,5213)]:
            rng = np.random.default_rng(case)
            face = rng.uniform(.878,.902)
            yaw = np.deg2rad(rng.uniform(-4.,4.))
            wall.body("wall").pos[0] = face+.2
            wall.key("spawn").qpos[3:7] = [np.cos(yaw/2),0,0,np.sin(yaw/2)]
            mujoco.mj_setConst(wall,mujoco.MjData(wall))
            mujoco.mj_saveModel(wall,scene)
            w = World(scene,"artifacts/encoder_actor.onnx")
            warm = w.run(command_schedule("push")[:50])
            assert not warm["other_robot_wall_impulse_Ns"].any() and not warm["contact_duty"].any()
            nominal = np.array([.42,0.,.01])
            approach = np.zeros((100,12),np.float32)
            approach[:,3:6] = spherical(np.linspace([.2,0,.1],nominal,100))
            common = w.run(approach)
            snap = w.snapshot()
            out = ROOT/str(case)
            out.mkdir(exist_ok=True)
            for label,tr in [("warm",warm),("approach",common)]:
                np.savez_compressed(out/f"{label}.npz",**tr)
            for arm in "ABC":
                w.restore(snap)
                u = np.tile(command(nominal),(N,1))
                level = 10*np.minimum(1.,np.arange(N)/50)
                if arm == "B":
                    u[:,6] = level
                if arm == "C":
                    u[:,3:6] = spherical(nominal+np.column_stack((level/200,np.zeros((N,2)))))
                tr = w.run(u)
                np.testing.assert_array_equal(tr["policy_command"],u)
                pre = np.vstack((common["qpos"][-1],tr["qpos"][:-1]))
                effective = np.vstack([ee_goal(p[None],c) for p,c in zip(pre,u)])
                projected = np.vstack([ee_goal(p[None],c*np.r_[np.ones(6),np.zeros(6)])
                                       for p,c in zip(pre,u)])
                if arm != "A":
                    point = nominal+np.column_stack((level/200,np.zeros((N,2))))
                    target = np.vstack([ee_goal(p[None],command(v)) for p,v in zip(pre,point)])
                    np.testing.assert_allclose(effective,target,atol=1e-7,rtol=0)
                np.savez_compressed(out/f"{arm}.npz",**tr,requested=u,nominal_pre=projected,
                                    effective_pre=effective)
                # Last 1.5 s at full command, retaining zero-load samples.
                x = tr["wall_normal_N"][-75:].ravel()
                loaded = x>1
                stops = np.flatnonzero(np.diff(np.r_[False,loaded,False]))
                longest = np.diff(stops.reshape(-1,2),axis=1).max(initial=0)*.005
                np.testing.assert_allclose(tr["wall_normal_N"].mean(1),tr["gripper_on_wall_N"][:,0])
                rows.append(dict(case=case,split="dev" if case<2 else "held",face_x_m=face,
                                 yaw_deg=np.rad2deg(yaw),arm=arm,mean_normal_N=x.mean(),
                                 peak_normal_N=x.max(),normal_sd_N=x.std(),impulse_Ns=.005*x.sum(),
                                 duty=loaded.mean(),longest_loaded_s=longest,
                                 effective_pre_rms_m=rms(length(tr["ee_world"][-75:]-effective[-75:])),
                                 other_Ns=tr["other_robot_wall_impulse_Ns"].sum(),
                                 max_tilt_deg=tilt(tr["qpos"]).max()))
    save(Path("docs/supplement_wall.csv"),rows)


if __name__ == "__main__":
    main()
