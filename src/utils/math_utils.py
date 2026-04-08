"""Math helpers used across simulation and SLAM."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def wrap_angle(theta: float) -> float:
    return (theta + math.pi) % (2 * math.pi) - math.pi


def rotation_matrix(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def pose_add(pose: np.ndarray, delta: np.ndarray) -> np.ndarray:
    rot = rotation_matrix(float(pose[2]))
    xy = pose[:2] + rot @ delta[:2]
    return np.array([xy[0], xy[1], wrap_angle(float(pose[2] + delta[2]))], dtype=float)


def local_to_world(pose: np.ndarray, point: np.ndarray) -> np.ndarray:
    return pose[:2] + rotation_matrix(float(pose[2])) @ point


def pairwise_mean(vectors: Iterable[np.ndarray]) -> np.ndarray:
    vectors = list(vectors)
    if not vectors:
        return np.zeros(2, dtype=float)
    return np.mean(np.stack(vectors, axis=0), axis=0)


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))
