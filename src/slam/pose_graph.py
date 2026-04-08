"""Lightweight pose-graph-like container."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PoseGraph:
    estimated_poses: list[np.ndarray] = field(default_factory=list)
    corrections: list[np.ndarray] = field(default_factory=list)

    def add_pose(self, pose: np.ndarray, correction: np.ndarray | None = None) -> None:
        self.estimated_poses.append(pose.copy())
        self.corrections.append(np.zeros(2, dtype=float) if correction is None else correction.copy())
