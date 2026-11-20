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
NODES = np.array([0., .25, .50])
TIMES = np.arange(25) * DT


def interpolate(knots, times):
    """Piecewise linear with a constant tail along the penultimate (knot) axis."""
    times = np.clip(times, NODES[0], NODES[-1])
    left = np.minimum(np.searchsorted(NODES, times, side="right") - 1, len(NODES) - 2)
    weight = ((times - NODES[left]) / (NODES[left + 1] - NODES[left]))[:, None]
    return knots[..., left, :] + weight * (knots[..., left + 1, :] - knots[..., left, :])


def sample(mean, sigma, winner, previous, scale, count, rng):
    knots = np.empty((count, *mean.shape))
    xyz = knots[..., :3]
    xyz[:] = mean[:, :3] + sigma[:, :3] * rng.standard_normal(xyz.shape)
    if knots.shape[-1] == 4:
        # One independent speed per trajectory, reached at knot 1 and held at knot 2.
        speed = mean[1, 3] + sigma[1, 3] * rng.standard_normal(count)
        knots[..., 3] = speed[:, None]
    knots[0] = winner
    knots[1] = mean
    np.clip(knots, -1, 1, out=knots)
    knots[:, 0] = np.clip(previous, -1, 1)
    for k, dt in enumerate(np.diff(NODES), start=1):
        delta = xyz[:, k] - xyz[:, k - 1]
        length = np.linalg.norm(delta * scale[:3], axis=-1, keepdims=True)
        xyz[:, k] = xyz[:, k - 1] + delta * np.minimum(1, .50 * dt / (length + 1e-12))
    if knots.shape[-1] == 4:
        max_change = .8 * (NODES[1] - NODES[0]) / scale[3]  # 0.8 m/s²
        knots[:, 1:, 3] = np.clip(knots[:, 1:2, 3], previous[3] - max_change,
                                  previous[3] + max_change)
    return knots


def fit(knots, costs):
    valid = np.flatnonzero(np.isfinite(costs))
    if not len(valid):
        raise RuntimeError("No valid candidate in this CEM batch")
    elite = valid[np.argsort(costs[valid], kind="stable")[:2]]
    return knots[elite].mean(0), knots[elite].std(0).clip(.12, .8), elite[0]
