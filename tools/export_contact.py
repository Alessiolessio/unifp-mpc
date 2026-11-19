"""Extract the local Spot USD gripper hulls in link coordinates and the joint stops."""

import argparse
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("../unifp-spot"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/spot_contact.npz"))
    args = parser.parse_args()
    stage = Usd.Stage.Open(str(args.source / "unifp_spot/assets/spot/usd/spot_with_arm.usd"))
    transforms = UsdGeom.XformCache()
    arrays = {}
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if prim.IsA(UsdPhysics.RevoluteJoint):
            joint = UsdPhysics.RevoluteJoint(prim)
            arrays["limit/" + prim.GetName()] = np.deg2rad(
                [joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()]
            )
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        body = prim.GetPath().pathString.split("/")[2]
        if body not in ("arm_link_fngr", "arm_link_wr1", "arm_link_jaw"):
            continue
        if prim.GetAttribute("physics:approximation").Get() != "convexHull":
            raise ValueError(f"Expected convex hull at {prim.GetPath()}")
        # The fixed jaw is part of wr1 in the MJCF. USD transforms multiply row vectors.
        owner = "arm_link_wr1" if body == "arm_link_jaw" else body
        link = stage.GetPrimAtPath("/spot_with_arm/" + owner)
        matrix = np.array(transforms.GetLocalToWorldTransform(prim)
                          * transforms.GetLocalToWorldTransform(link).GetInverse())
        points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get())
        arrays[owner + "/" + prim.GetParent().GetName()] = points @ matrix[:3, :3] + matrix[3, :3]
    if len(arrays) != 19 + 8:
        raise ValueError("Expected 19 joint limits and eight wrist/jaw/finger hulls")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **arrays)
    print(args.output)


if __name__ == "__main__":
    main()
