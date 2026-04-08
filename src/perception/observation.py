"""Observation datamodels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Observation:
    object_id: str
    label: str
    world_position: np.ndarray
    relative_position: np.ndarray
    confidence: float
    radius: float
    visit_idx: int
    step_idx: int


@dataclass
class FrameData:
    visit_idx: int
    step_idx: int
    true_pose: np.ndarray
    odom_delta: np.ndarray
    observations: list[Observation]
