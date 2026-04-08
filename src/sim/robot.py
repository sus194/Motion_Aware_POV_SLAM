"""Robot state and odometry simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.utils.math_utils import rotation_matrix, wrap_angle


@dataclass
class RobotConfig:
    sensor_range: float
    fov_deg: float
    odom_noise_xy: float
    odom_noise_theta: float


def simulate_odometry(poses: np.ndarray, cfg: RobotConfig, rng: np.random.Generator) -> np.ndarray:
    """Return noisy local-frame odometry deltas for a ground-truth pose sequence."""
    deltas = []
    for idx in range(1, len(poses)):
        prev = poses[idx - 1]
        curr = poses[idx]
        global_delta_xy = curr[:2] - prev[:2]
        local_delta_xy = rotation_matrix(-float(prev[2])) @ global_delta_xy
        delta_theta = wrap_angle(float(curr[2] - prev[2]))

        noisy = np.array([local_delta_xy[0], local_delta_xy[1], delta_theta], dtype=float)
        noisy[:2] += rng.normal(0.0, cfg.odom_noise_xy, size=2)
        noisy[2] += float(rng.normal(0.0, cfg.odom_noise_theta))
        noisy[2] = wrap_angle(float(noisy[2]))
        deltas.append(noisy)
    return np.stack(deltas, axis=0) if deltas else np.zeros((0, 3), dtype=float)
