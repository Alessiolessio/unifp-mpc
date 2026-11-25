"""One development-only ReLIC ablation on the two runtime calibration cones."""
import json
import sys
from unittest.mock import patch

import numpy as np
from sumo.tasks.spot.spot_cone_upright import SpotConeUpright
from sumo.tasks.spot.spot_upright import z_axis_orientation_reward

from supplement_sumo import ROOT, worker


def release_reward(self, states, sensors, controls, system_metadata=None):
    axis = sensors[..., self.object_z_axis_idx:self.object_z_axis_idx+3]
    i = self.end_effector_to_object_idx
    gap = np.linalg.norm(sensors[..., i:i+3], axis=-1)
    # Same distance term before tipping. Above the native upright threshold,
    # distance beyond 30 cm earns no additional reward.
    distance = np.where(axis[..., 2] < .9, gap, np.maximum(.30-gap, 0.))
    return z_axis_orientation_reward(self.config,axis)-self.config.w_gripper_proximity*distance.mean(-1)


if __name__ == "__main__":
    case = sys.argv[1]
    assert case in ("near-0", "general-82")
    control = ROOT/"dev/cone"/case/"24x2"
    spec = json.loads((control/"assignment.json").read_text())
    out = ROOT/"release_dev"/case
    with patch.object(SpotConeUpright, "reward", release_reward):
        worker(spec,out)
    with np.load(control/"episode.npz") as a, np.load(out/"episode.npz") as b:
        np.testing.assert_array_equal(a["initial"],b["initial"])
