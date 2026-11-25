"""External task outcomes at common 50 Hz post-action ticks and 100 ms score boundaries."""
import numpy as np


def tilt(q):
    w, x, y, z = q[:, 3:7].T
    return np.rad2deg(np.maximum(abs(np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y))),
                                 abs(np.arcsin(np.clip(2*(w*y-z*x), -1, 1)))))


def score(task, q, v, touch, ground, other, mass, initial, goal=None):
    """Exclude warmup. Forces are substep means, touch is any substep, other in N s."""
    n = len(q)
    if not n:
        q = initial[None]
    posture = tilt(q)
    unsafe = (posture >= 30) | (q[:, 2] < .35)
    result = dict(max_tilt_deg=float(posture.max()), min_height_m=float(q[:, 2].min()),
                  other_Ns=float(other.sum()), strict=False, safe_strict=False, settled_s=None)
    if task == "cone":
        alignment = np.clip(1-2*(q[:, -3]**2+q[:, -2]**2), -1, 1)
        angle = np.rad2deg(np.arccos(alignment))
        native = np.flatnonzero(alignment[:n] >= .9)
        good = ((angle[:n] <= 10) & (np.linalg.norm(v[:, -3:], axis=1) <= .25)
                & (ground >= .5*mass*9.81) & ~touch & ~unsafe[:n])
        result.update(final_error=float(angle[-1]), upright_s=(int(native[0])+1)*.02 if len(native) else None,
                      min_tilt_deg=float(angle.min()), final_support_N=float(ground[-1]) if n else None)
    else:
        d = goal-initial[-7:-5]
        u = d/np.linalg.norm(d)
        residual = q[:, -7:-5]-goal
        err = np.linalg.norm(residual, axis=1)
        along = residual @ u
        speed = np.linalg.norm(v[:, -6:-4], axis=1)
        good = err[:n] <= .015
        result.update(final_error=float(err[-1]), overshoot_m=float(max(0., along.max())),
                      undershoot_m=float(max(0., -along[-1])),
                      cross_track_m=float(abs(residual @ [-u[1], u[0]]).max()))
    for end in range(25, n+1, 5):
        if good[end-25:end].all() and (task == "cone" or speed[end-5:end].max() <= .02):
            result.update(strict=True, settled_s=.02*end,
                          safe_strict=not (unsafe[:end].any() or other[:end].any()))
            break
    return result
