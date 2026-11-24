"""Frame conversions used by the planner and offline telemetry."""
import numpy as np


def yaw_of(quat):
    w, x, y, z = np.moveaxis(quat, -1, 0)
    return np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))


def world_ee_target(root, spherical_command):
    """Decode the issued spherical command using the pre-action root pose."""
    r, pitch, az = np.moveaxis(spherical_command, -1, 0)
    yaw = yaw_of(root[..., 3:7])
    return root[..., :3] + np.stack((.35*np.cos(yaw) + r*np.cos(pitch)*np.cos(az+yaw),
                                     .35*np.sin(yaw) + r*np.cos(pitch)*np.sin(az+yaw),
                                     .12+r*np.sin(pitch)), axis=-1)


def show_cone_mesh(model):
    obj = model.body("traffic_cone").id
    model.geom_group[(model.geom_bodyid == obj) & (model.geom_contype != 0)] = 3
    model.site_rgba[model.site_bodyid == obj, 3] = 0.


def cone_axis(quat):
    """World direction of body +Z from a MuJoCo unit quaternion (wxyz)."""
    w, x, y, z = np.moveaxis(quat, -1, 0)
    return np.stack((2*(x*z + w*y), 2*(y*z - w*x), 1-2*(x*x + y*y)), axis=-1)


def angle_deg(quat):
    return np.rad2deg(np.arccos(np.clip(cone_axis(quat)[..., 2], -1., 1.)))


def nominal_targets(root, raw):
    dx, dy, dz = np.moveaxis(raw-root[..., :3], -1, 0)
    yaw = yaw_of(root[..., 3:7])
    c, s = np.cos(yaw), np.sin(yaw)
    x, y, z = c*dx+s*dy-.35, -s*dx+c*dy, dz-.12
    r = np.sqrt(x*x+y*y+z*z)
    spherical = np.stack((r, np.arcsin(z/(r+1e-8)), np.arctan2(y, x)), axis=-1)
    spherical = spherical.clip([.15, np.deg2rad(-40), -np.pi/2], [.55, np.deg2rad(50), np.pi/2])
    return world_ee_target(root, spherical)


def world_force(root, cmd):
    yaw = yaw_of(root[..., 3:7])
    c, s = np.cos(yaw), np.sin(yaw)
    x, y = cmd[..., 6], cmd[..., 7]
    return np.stack((c*x-s*y, s*x+c*y, cmd[..., 8]), axis=-1)
