"""Trajectory generation."""

from __future__ import annotations

import math

import numpy as np

from src.utils.math_utils import wrap_angle


def generate_lawnmower_path(width: float, height: float, steps: int, margin: float = 1.5) -> np.ndarray:
    x_left = margin
    x_right = width - margin
    stripes = 5
    ys = np.linspace(margin, height - margin, stripes)

    waypoints: list[np.ndarray] = []
    for idx, y in enumerate(ys):
        x_target = x_right if idx % 2 == 0 else x_left
        waypoints.append(np.array([x_target, y], dtype=float))
    waypoints.insert(0, np.array([x_left, ys[0]], dtype=float))

    path = []
    segment_steps = max(2, steps // max(1, len(waypoints) - 1))
    for start, end in zip(waypoints[:-1], waypoints[1:]):
        for alpha in np.linspace(0.0, 1.0, segment_steps, endpoint=False):
            xy = (1 - alpha) * start + alpha * end
            theta = math.atan2(end[1] - start[1], end[0] - start[0])
            path.append(np.array([xy[0], xy[1], wrap_angle(theta)], dtype=float))
    path.append(np.array([waypoints[-1][0], waypoints[-1][1], path[-1][2]], dtype=float))
    return np.stack(path[:steps], axis=0)
