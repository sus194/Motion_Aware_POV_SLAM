"""Landmark management for baseline and motion-aware SLAM."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RelocationCandidate:
    position: np.ndarray
    confirmations: int = 1


@dataclass
class Landmark:
    landmark_id: str
    label: str
    position: np.ndarray
    observation_count: int = 1
    last_residual: float = 0.0
    state_label: str = "STATIC"
    relocation_candidate: RelocationCandidate | None = None
    gt_object_id: str | None = None


@dataclass
class LandmarkManager:
    update_alpha: float
    landmarks: dict[str, Landmark] = field(default_factory=dict)
    next_id: int = 0
    false_updates: int = 0

    def candidates(self) -> list[tuple[str, str, np.ndarray]]:
        return [(landmark_id, landmark.label, landmark.position.copy()) for landmark_id, landmark in self.landmarks.items()]

    def add(self, label: str, position: np.ndarray, gt_object_id: str | None = None) -> str:
        landmark_id = f"lm_{self.next_id:03d}"
        self.next_id += 1
        self.landmarks[landmark_id] = Landmark(landmark_id=landmark_id, label=label, position=position.copy(), gt_object_id=gt_object_id)
        return landmark_id

    def update_position(self, landmark_id: str, measured_position: np.ndarray, residual: float) -> None:
        landmark = self.landmarks[landmark_id]
        landmark.position = (1.0 - self.update_alpha) * landmark.position + self.update_alpha * measured_position
        landmark.observation_count += 1
        landmark.last_residual = residual

    def consider_relocation(self, landmark_id: str, new_position: np.ndarray, relocation_radius: float, relocation_confirmations: int, gt_object_id: str | None) -> bool:
        landmark = self.landmarks[landmark_id]
        candidate = landmark.relocation_candidate
        if candidate is None or float(np.linalg.norm(candidate.position - new_position)) > relocation_radius:
            landmark.relocation_candidate = RelocationCandidate(position=new_position.copy(), confirmations=1)
            return False
        candidate.confirmations += 1
        if candidate.confirmations >= relocation_confirmations:
            if gt_object_id is not None and landmark.gt_object_id is not None and gt_object_id != landmark.gt_object_id:
                self.false_updates += 1
            landmark.position = candidate.position.copy()
            landmark.relocation_candidate = None
            landmark.observation_count += 1
            return True
        return False
