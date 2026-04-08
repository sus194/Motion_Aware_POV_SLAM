"""Simple deterministic nearest-neighbor data association."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from src.perception.observation import Observation


@dataclass
class AssociationResult:
    observation: Observation | None
    match_id: str | None
    distance: float


def associate_position(label: str, position: np.ndarray, candidates: Iterable[tuple[str, str, np.ndarray]], gate: float) -> AssociationResult:
    best_id: str | None = None
    best_dist = float("inf")
    for candidate_id, candidate_label, candidate_position in candidates:
        if candidate_label != label:
            continue
        dist = float(np.linalg.norm(candidate_position - position))
        if dist < best_dist and dist <= gate:
            best_id = candidate_id
            best_dist = dist
    return AssociationResult(observation=None, match_id=best_id, distance=best_dist)


def associate_nearest(observation: Observation, candidates: Iterable[tuple[str, str, np.ndarray]], gate: float) -> AssociationResult:
    result = associate_position(observation.label, observation.world_position, candidates, gate)
    result.observation = observation
    return result
