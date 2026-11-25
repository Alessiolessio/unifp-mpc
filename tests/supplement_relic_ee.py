"""Post-action ReLIC finger and body velocities from recorded state, without a new solve."""
from pathlib import Path
from tempfile import TemporaryDirectory

import judo.tasks.base as base
import mujoco
import numpy as np
from sumo.tasks.spot.spot_box_push import SpotBoxPush
from sumo.tasks.spot.spot_cone_upright import SpotConeUpright

from supplement_sumo import ROOT


def main():
    with TemporaryDirectory() as tmp:
        (Path(tmp)/"meshes").symlink_to(base.MODEL_PATH/"meshes",target_is_directory=True)
        base.MODEL_PATH = Path(tmp)
        for task, ctor in [("box",SpotBoxPush),("cone",SpotConeUpright)]:
            model = ctor().model
            data = mujoco.MjData(model)
            finger = model.body("arm_link_fngr").id
            for path in sorted((ROOT/"paired").glob("*/*/episode.npz")):
                if (int(path.parents[1].name)<3100) != (task=="box"):
                    continue
                with np.load(path) as e:
                    q, v = e["qpos"],e["qvel"]
                assert (q.shape[1],v.shape[1]) == (model.nq,model.nv)
                ee, root = [], []
                for pose,speed in zip(q,v,strict=True):
                    data.qpos[:],data.qvel[:] = pose,speed
                    mujoco.mj_kinematics(model,data)
                    ee.append(data.xpos[finger].copy())
                    rotation = data.xmat[model.body("body").id].reshape(3,3)
                    # MuJoCo free-joint translation is world, angular velocity is local.
                    root.append(np.r_[rotation.T@speed[:3],speed[3:6]])
                np.savez_compressed(path.parent/"kinematics.npz",finger_world_m=ee,
                                    root_linear_angular_body=root)


if __name__=="__main__":
    main()
