"""Motion feature estimation for state classification."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np


@dataclass
class MotionFeatures:
    avg_displacement: float
    avg_residual: float
    observation_count: int
    revisit_jump: float


class MotionFeatureBank:
    def __init__(self, maxlen: int = 12) -> None:
        self.positions: dict[str, deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=maxlen))
        self.residuals: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=maxlen))
        self.visit_positions: dict[str, dict[int, list[np.ndarray]]] = defaultdict(dict)

    def update(self, object_id: str, visit_idx: int, position: np.ndarray, residual: float) -> MotionFeatures:
        self.positions[object_id].append(position.copy())
        self.residuals[object_id].append(float(residual))
        self.visit_positions.setdefault(object_id, {}).setdefault(visit_idx, []).append(position.copy())
        return self.compute(object_id)

    def compute(self, object_id: str) -> MotionFeatures:
        positions = list(self.positions.get(object_id, []))
        residuals = list(self.residuals.get(object_id, []))
        if len(positions) >= 2:
            displacements = [float(np.linalg.norm(b - a)) for a, b in zip(positions[:-1], positions[1:])]
            avg_displacement = float(np.mean(displacements))
        else:
            avg_displacement = 0.0

        revisit_means = []
        for visit_idx in sorted(self.visit_positions.get(object_id, {})):
            points = self.visit_positions[object_id][visit_idx]
            revisit_means.append(np.mean(np.stack(points, axis=0), axis=0))
        if len(revisit_means) >= 2:
            revisit_jump = float(np.linalg.norm(revisit_means[-1] - revisit_means[-2]))
        else:
            revisit_jump = 0.0

        return MotionFeatures(
            avg_displacement=avg_displacement,
            avg_residual=float(np.mean(residuals)) if residuals else 0.0,
            observation_count=len(positions),
            revisit_jump=revisit_jump,
        )
