"""Heuristic motion-state classifier."""

from __future__ import annotations

from dataclasses import dataclass

from src.tracking.motion_estimator import MotionFeatures


@dataclass
class StateProbabilities:
    static: float
    semi_static: float
    dynamic: float


class MotionStateClassifier:
    def __init__(self, dynamic_velocity_threshold: float, semi_static_revisit_threshold: float, persistence_threshold: int) -> None:
        self.dynamic_velocity_threshold = dynamic_velocity_threshold
        self.semi_static_revisit_threshold = semi_static_revisit_threshold
        self.persistence_threshold = persistence_threshold

    def predict(self, features: MotionFeatures) -> tuple[str, StateProbabilities]:
        if features.observation_count >= 2 and features.avg_displacement >= self.dynamic_velocity_threshold and features.avg_residual >= 0.35:
            return "DYNAMIC", StateProbabilities(static=0.05, semi_static=0.15, dynamic=0.80)
        if features.observation_count >= self.persistence_threshold and features.revisit_jump >= self.semi_static_revisit_threshold:
            return "SEMI_STATIC", StateProbabilities(static=0.20, semi_static=0.65, dynamic=0.15)
        return "STATIC", StateProbabilities(static=0.80, semi_static=0.15, dynamic=0.05)
