"""Two prespecified fixed-command diagnostics independent of the task-level MPC comparisons."""
import csv
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np

from fixed_finger_pulse import run as pulse
from fixed_policy_loads import N, MODES, command, ee_goal, check_workspace, prepare_case
from final_scenes import POLICY, ROOT
from physical_score import score
from unifp_native import World


def push(case, scene):
    _, warm, xyz, vel, meta = prepare_case("box", case, False, scene, 1.)
    live = World(str(scene), POLICY)
    w = live.run(warm)
    assert not w["contact_duty"].any() and not w["other_robot_box_impulse_Ns"].any()
    s = live.snapshot()
    start = w["qpos"][-1]
    goal = start[-7:-5]+[.05,0]
    a = command(xyz, vel)
    b, c = a.copy(), command(xyz+[.06,0,0], vel)
    b[6] = 12.
    np.testing.assert_allclose(ee_goal(start[None],b),ee_goal(start[None],c), atol=1e-6, rtol=0)
    for arm, cmd in zip(MODES,(a,b,c),strict=True):
        check_workspace(cmd)
        live.restore(s)
        tr = live.run(np.tile(cmd,(N,1)))
        result = score("box", tr["qpos"], tr["qvel"], tr["contact_duty"]>0,
                       tr["ground_on_box_N"][:,2], tr["other_robot_box_impulse_Ns"],1.,start,goal)
        end_error = np.linalg.norm(tr["qpos"][-25:,-7:-5]-goal,axis=1)
        end_speed = np.linalg.norm(tr["qvel"][-5:,-6:-4],axis=1)
        yield dict(case=case, arm=arm, **meta, **result,
                   settled_at_end=bool(end_error.max() <= .015 and end_speed.max() <= .02))


def save(path, rows):
    with path.open("w") as f:
        w = csv.DictWriter(f,rows[0],lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    with TemporaryDirectory(dir=ROOT) as tmp:
        m = mujoco.MjModel.from_binary_path("artifacts/spot.mjb")
        rows = []
        for case in [0,1,*range(4201,4213)]:
            rows.extend(pulse(case,m,Path(tmp),synchronized=True))
        save(Path("docs/final_pulse.csv"), rows)
        rows = []
        for case in range(4101,4113):
            rows.extend(push(case,Path(tmp)/"box.mjb"))
        save(Path("docs/final_fixed_goal.csv"),rows)
