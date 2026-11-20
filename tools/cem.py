"""World-position knot sampling and elite fitting adapted from Judo's MIT-licensed CEM."""
# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np

DT = .02
PREFIX = 5
BASE_SLEW = .8  # m/s² for translation, rad/s² for yaw
NODES = np.array([0., .25, .50])
TIMES = np.arange(25) * DT


def interpolate(knots, times, nodes=NODES):
    """Piecewise linear with a constant tail along the penultimate (knot) axis."""
    times = np.clip(times, nodes[0], nodes[-1])
    left = np.minimum(np.searchsorted(nodes, times, side="right") - 1, len(nodes) - 2)
    weight = ((times - nodes[left]) / (nodes[left + 1] - nodes[left]))[:, None]
    return knots[..., left, :] + weight * (knots[..., left + 1, :] - knots[..., left, :])


def sample(mean, sigma, winner, previous, scale, count, rng, nodes=NODES):
    knots = np.empty((count, *mean.shape))
    xyz = knots[..., :3]
    xyz[:] = mean[:, :3] + sigma[:, :3] * rng.standard_normal(xyz.shape)
    for axis in range(3, knots.shape[-1]):
        # One speed per body axis, reached at knot 1 and held at knot 2.
        speed = mean[1, axis] + sigma[1, axis] * rng.standard_normal(count)
        knots[..., axis] = speed[:, None]
    knots[0] = winner
    knots[1] = mean
    np.clip(knots, -1, 1, out=knots)
    knots[:, 0] = np.clip(previous, -1, 1)
    for k, dt in enumerate(np.diff(nodes), start=1):
        delta = xyz[:, k] - xyz[:, k - 1]
        length = np.linalg.norm(delta * scale[:3], axis=-1, keepdims=True)
        xyz[:, k] = xyz[:, k - 1] + delta * np.minimum(1, .50 * dt / (length + 1e-12))
    for axis in range(3, knots.shape[-1]):
        max_change = BASE_SLEW * (nodes[1] - nodes[0]) / scale[axis]
        knots[:, 1:, axis] = np.clip(knots[:, 1:2, axis], previous[axis] - max_change,
                                     previous[axis] + max_change)
    return knots


def fit(knots, costs, elites=2):
    valid = np.flatnonzero(np.isfinite(costs))
    if not len(valid):
        raise RuntimeError("No valid candidate in this CEM batch")
    elite = valid[np.argsort(costs[valid], kind="stable")[:elites]]
    return knots[elite].mean(0), knots[elite].std(0).clip(.12, .8), elite[0]
