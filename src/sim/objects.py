"""Simulation objects and motion behavior."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class MotionType(str, Enum):
    STATIC = "STATIC"
    SEMI_STATIC = "SEMI_STATIC"
    DYNAMIC = "DYNAMIC"


@dataclass
class DynamicMotion:
    pattern: str
    velocity: np.ndarray
    waypoints: list[np.ndarray] = field(default_factory=list)
    smoothing: float = 0.85


@dataclass
class SimObject:
    object_id: str
    label: str
    radius: float
    motion_type: MotionType
    base_position: np.ndarray
    visit_positions: dict[int, np.ndarray] = field(default_factory=dict)
    dynamic_motion: DynamicMotion | None = None

    def position_at(self, visit_idx: int, step_idx: int, bounds: tuple[float, float]) -> np.ndarray:
        if self.motion_type == MotionType.STATIC:
            return self.base_position.copy()
        if self.motion_type == MotionType.SEMI_STATIC:
            return self.visit_positions.get(visit_idx, self.base_position).copy()
        if self.dynamic_motion is None:
            return self.base_position.copy()
        if self.dynamic_motion.pattern == "straight":
            raw = self.base_position + step_idx * self.dynamic_motion.velocity
            return np.array([
                np.clip(raw[0], self.radius, bounds[0] - self.radius),
                np.clip(raw[1], self.radius, bounds[1] - self.radius),
            ], dtype=float)
        if self.dynamic_motion.pattern == "waypoint" and self.dynamic_motion.waypoints:
            waypoint = self.dynamic_motion.waypoints[step_idx % len(self.dynamic_motion.waypoints)]
            alpha = min(1.0, step_idx / max(1, len(self.dynamic_motion.waypoints)))
            return (1.0 - alpha) * self.base_position + alpha * waypoint
        angle = 0.35 * step_idx
        wiggle = np.array([
            self.dynamic_motion.velocity[0] * np.cos(angle),
            self.dynamic_motion.velocity[1] * np.sin(angle),
        ], dtype=float)
        raw = self.base_position + step_idx * 0.4 * wiggle
        return np.array([
            np.clip(raw[0], self.radius, bounds[0] - self.radius),
            np.clip(raw[1], self.radius, bounds[1] - self.radius),
        ], dtype=float)


def label_radius(label: str) -> float:
    return {
        "shelves": 0.9,
        "boxes": 0.5,
        "carts": 0.65,
        "cones": 0.35,
        "pillars": 0.75,
    }.get(label, 0.5)


def object_to_dict(obj: SimObject) -> dict[str, Any]:
    return {
        "object_id": obj.object_id,
        "label": obj.label,
        "radius": obj.radius,
        "motion_type": obj.motion_type.value,
        "base_position": obj.base_position.tolist(),
        "visit_positions": {str(k): v.tolist() for k, v in obj.visit_positions.items()},
    }
