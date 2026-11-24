"""Compare one recording to an archived episode; check forecast times and measured prefixes."""
import argparse
from pathlib import Path
import numpy as np


def check(folder, reference):
    with np.load(folder/"episode.npz") as e, np.load(reference/"episode.npz") as old:
        for k in ("qpos", "qvel", "action", "policy_command", "ee_world", "requests", "targets",
                  "gripper_on_cone_N", "gripper_on_box_N", "gripper_on_wall_N", "tension_N"):
            if k in old:
                np.testing.assert_array_equal(e[k], old[k])
        starts = e["forecast_start_tick"]
        np.testing.assert_array_equal(np.diff(starts), np.full(len(starts)-1, 5))
        roots = e["forecast_pre_root"]
        previous = np.maximum(0, starts-1)
        valid = starts > 0
        np.testing.assert_array_equal(roots[valid], e["qpos"][previous[valid], :7])
        if not valid[0]:
            np.testing.assert_array_equal(roots[0], e["initial_qpos"][:7])
        ticks = starts[:, None]+np.arange(5)
        modeled = e["exact"][ticks] if "exact" in e else np.ones(ticks.shape, bool)
        for k in e.files:
            if k == "forecast_reference_xy":
                # Reference time is computed by horizon addition and by absolute tick.
                np.testing.assert_allclose(e[k][:, :5], e["reference_xy"][ticks], rtol=0, atol=1e-12)
            elif k.startswith("forecast_") and k[9:] in e:
                np.testing.assert_array_equal(e[k][:, :5][modeled], e[k[9:]][ticks][modeled])
        if "requests" in e:
            np.testing.assert_array_equal(e["forecast_request"][:, :5].reshape(e["requests"].shape),
                                          e["requests"])
        print(folder, len(starts), "winning horizons;", modeled.sum(), "modeled ticks exact")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("recording", type=Path)
    p.add_argument("reference", type=Path)
    a = p.parse_args()
    check(a.recording, a.reference)
