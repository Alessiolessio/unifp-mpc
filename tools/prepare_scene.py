"""Compile local Spot MJCFs with the UniFP controller parameters into MuJoCo 3.6 binaries."""

import argparse
import runpy
from pathlib import Path

import mujoco
import numpy as np


JOINTS = ("arm_sh0 fl_hx fr_hx hl_hx hr_hx arm_sh1 fl_hy fr_hy hl_hy hr_hy "
          "arm_el0 fl_kn fr_kn hl_kn hr_kn arm_el1 arm_wr0 arm_wr1 arm_f1x").split()
DEFAULT = [0, .1, -.1, .1, -.1, -.9, .9, .9, 1.1, 1.1, 1.8, -1.5, -1.5, -1.5, -1.5,
           0, -.9, 0, -1.54]
# Per-group lags from the controlled seed-0 Isaac reset, in 5 ms physics steps.
ARM_LAGS = [2, 0, 3, 0, 1, 1, 3]


def compile_scene(judo, menagerie, constants, contact, box):
    scene = "spot_box" if box else "spot_primitive"
    spec = mujoco.MjSpec.from_file(str(judo / "judo/models/xml" / scene / "robot.xml"))
    for mesh in spec.meshes:
        mesh.file = str(menagerie / "assets" / Path(mesh.file).name)
    for texture in spec.textures:
        if texture.file:
            texture.file = str(menagerie / "spot.png")

    spec.option.timestep = .005
    spec.option.density = 0
    spec.body("body").pos = [0, 0, .5]
    spec.geom("ground").friction = [1, .005, .0001]
    # At reset the front feet penetrate the plane by 19 mm. A 10 ms response
    # better reproduces the measured PhysX transient at the 5 ms step size.
    spec.geom("ground").solref[0] = .01
    spec.joint("arm_f1x").solref_limit[0] = .01
    for body_name in ("arm_link_fngr", "arm_link_wr1"):
        body = spec.body(body_name)
        for geom in list(body.geoms):
            if geom.contype or geom.conaffinity:
                spec.delete(geom)
    for key in contact.files:
        owner, name = key.split("/")
        if owner == "limit":
            spec.joint(name).range = contact[key]
        else:
            mesh = spec.add_mesh(name="usd_" + name, uservert=contact[key].ravel())
            spec.body(owner).add_geom(spec.find_default("collision"), name=mesh.name,
                                      type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh.name)
    table = np.array(constants["JOINT_PARAMETER_LOOKUP_TABLE"])[:, [0, 2]]
    spec.add_numeric(name="knee_angle_limit", data=table.ravel())
    for motor in spec.actuators:
        name = motor.target
        i = JOINTS.index(name)
        joint = spec.joint(name)
        joint.damping = joint.frictionloss = 0
        if name.startswith("arm_"):
            a = constants["ARM_JOINT_NAMES"].index(name)
            kp = constants["ARM_STIFFNESS"][a]
            kd = constants["ARM_DAMPING"][a]
            limit = constants["ARM_EFFORT_LIMIT"][a]
            lag = ARM_LAGS[a]
            joint.armature = constants["ARM_ARMATURE"][a]
        else:
            kp, kd = 120, 3
            knee = name.endswith("_kn")
            limit = 0 if knee else constants["HIP_EFFORT_LIMIT"]
            lag = 2 if knee else 0
            joint.armature = 0
        motor.set_to_motor()
        motor.ctrllimited = motor.forcelimited = False
        motor.userdata = [i, DEFAULT[i], kp, kd, limit, lag]

    if box:
        # Use the SUMO/Judo free-box topology and the measured Isaac comparison object's
        # dimensions and uniform-density cuboid inertia. Its min friction rule selects 0.30.
        body = spec.body("box_body")
        body.pos = [1.08, 0, .252]
        body.mass = 1
        body.inertia = [(0.4**2 + 0.5**2) / 12] * 2 + [2 * 0.4**2 / 12]
        for name in ("box_visual", "box_collision"):
            spec.geom(name).size = [.20, .20, .25]
        geom = spec.geom("box_collision")
        geom.priority = 6
        geom.friction = [.3, .005, .0001]

    spec.add_key(name="spawn")
    model = spec.compile()
    key = model.key("spawn").qpos
    key[:] = model.qpos0
    for name, angle in zip(JOINTS, DEFAULT, strict=True):
        key[model.joint(name).qposadr] = angle
    if box:
        for name, angle in zip(("arm_sh1", "arm_el0", "arm_wr0"), (-1.5, 3, -1.5), strict=True):
            key[model.joint(name).qposadr] = angle
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judo", type=Path, default=Path("../judo"))
    parser.add_argument("--source", type=Path, default=Path("../unifp-spot"))
    parser.add_argument("--menagerie", type=Path, default=Path.home() /
                        ".cache/robot_descriptions/mujoco_menagerie/boston_dynamics_spot")
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    constants = runpy.run_path(str(args.source / "unifp_spot/assets/spot/constants.py"))
    args.out.mkdir(parents=True, exist_ok=True)
    with np.load(args.out / "spot_contact.npz") as contact:
        for box, name in ((False, "spot"), (True, "spot_box")):
            model = compile_scene(args.judo, args.menagerie.resolve(), constants, contact, box)
            path = args.out / f"{name}.mjb"
            mujoco.mj_saveModel(model, str(path))
            print(f"{path}  {model.nu} torque motors, {model.opt.timestep:g} s")


if __name__ == "__main__":
    main()
