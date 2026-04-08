"""POV-SLAM-style object-aware semi-static baseline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.perception.data_association import associate_position
from src.perception.observation import FrameData
from src.slam.landmark_manager import LandmarkManager
from src.slam.optimizer import weighted_pose_correction
from src.slam.pose_graph import PoseGraph
from src.utils.math_utils import local_to_world, pose_add


@dataclass
class SlamRunResult:
    name: str
    estimated_poses: list[np.ndarray]
    landmark_positions: dict[str, np.ndarray]
    landmark_states: dict[str, str]
    landmark_truth: dict[str, str | None]
    residual_history: list[float]
    runtime_per_frame: list[float]
    false_landmark_updates: int = 0
    track_stats: dict[str, float] = field(default_factory=dict)
    frame_debug: list[dict[str, Any]] = field(default_factory=list)
    object_state_estimates: dict[str, str] = field(default_factory=dict)


class BaselineObjectSLAM:
    """Simplified POV-SLAM baseline for semi-static object consistency.

    This approximates the paper's semi-static object handling without dynamic-object
    state: observations get residual-based consistency weights, and persistent
    inconsistent observations can relocate landmarks instead of becoming tracks.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        slam_cfg = config["slam"]
        self.landmarks = LandmarkManager(update_alpha=float(slam_cfg.get("baseline_landmark_update_alpha", slam_cfg["landmark_update_alpha"])))
        self.pose_graph = PoseGraph()
        self.association_distance = float(slam_cfg.get("baseline_association_distance", slam_cfg["association_distance"]))
        self.outlier_distance = float(slam_cfg.get("baseline_outlier_distance", slam_cfg["outlier_distance"]))
        self.pose_correction_gain = float(slam_cfg.get("baseline_pose_correction_gain", slam_cfg["pose_correction_gain"]))
        self.relocation_confirmations = int(slam_cfg.get("relocation_confirmations", 3))
        self.relocation_radius = float(slam_cfg.get("relocation_radius", 1.0))
        self.consistency_sigma = float(slam_cfg.get("pov_consistency_sigma", 1.2))
        self.reliable_weight_threshold = float(slam_cfg.get("pov_reliable_weight_threshold", 0.35))
        self.visit_initial_noise_xy = float(slam_cfg.get("visit_initial_noise_xy", 0.0))
        self.visit_initial_noise_theta = float(slam_cfg.get("visit_initial_noise_theta", 0.0))
        self.runtime_per_frame: list[float] = []
        self.residual_history: list[float] = []
        self.current_pose = np.zeros(3, dtype=float)
        self.frame_debug: list[dict[str, Any]] = []
        self.object_state_estimates: dict[str, str] = {}
        self.initialized = False
        self.current_visit: int | None = None

    def _initial_pose_for_visit(self, frame: FrameData) -> np.ndarray:
        pose = frame.true_pose.copy()
        if frame.visit_idx > 0:
            sign = -1.0 if frame.visit_idx % 2 else 1.0
            pose[0] += sign * self.visit_initial_noise_xy
            pose[1] -= 0.6 * self.visit_initial_noise_xy
            pose[2] += sign * self.visit_initial_noise_theta
        return pose

    def _consistency_weight(self, residual: float, confidence: float) -> float:
        gaussian_part = float(np.exp(-0.5 * (residual / max(1e-6, self.consistency_sigma)) ** 2))
        return confidence * gaussian_part

    def _process_frame(self, frame: FrameData) -> None:
        if (not self.initialized) or (self.current_visit != frame.visit_idx):
            self.current_pose = self._initial_pose_for_visit(frame)
            self.initialized = True
            self.current_visit = frame.visit_idx
        else:
            self.current_pose = pose_add(self.current_pose, frame.odom_delta)

        residuals: list[np.ndarray] = []
        weights: list[float] = []

        for obs in frame.observations:
            measured_world = local_to_world(self.current_pose, obs.relative_position)
            association = associate_position(obs.label, measured_world, self.landmarks.candidates(), self.association_distance)
            if association.match_id is None:
                self.landmarks.add(obs.label, measured_world, gt_object_id=obs.object_id)
                self.object_state_estimates[obs.object_id] = "STATIC"
                continue

            landmark = self.landmarks.landmarks[association.match_id]
            anchor = landmark.position.copy()
            landmark_residual_vec = measured_world - anchor
            landmark_residual = float(np.linalg.norm(landmark_residual_vec))
            self.residual_history.append(landmark_residual)
            consistency = self._consistency_weight(landmark_residual, obs.confidence)

            if consistency >= self.reliable_weight_threshold and landmark_residual <= self.outlier_distance:
                residuals.append(anchor - measured_world)
                weights.append(consistency)
                self.landmarks.update_position(association.match_id, measured_world, landmark_residual)
                self.object_state_estimates[obs.object_id] = "STATIC"
            else:
                self.landmarks.consider_relocation(
                    association.match_id,
                    measured_world,
                    self.relocation_radius,
                    self.relocation_confirmations,
                    obs.object_id,
                )
                self.object_state_estimates[obs.object_id] = "SEMI_STATIC"

        correction = weighted_pose_correction(residuals, weights, self.pose_correction_gain)
        self.current_pose[:2] += correction
        self.pose_graph.add_pose(self.current_pose, correction)
        self.frame_debug.append(
            {
                "visit_idx": frame.visit_idx,
                "step_idx": frame.step_idx,
                "estimated_pose": self.current_pose.copy().tolist(),
                "landmarks": {landmark_id: landmark.position.copy().tolist() for landmark_id, landmark in self.landmarks.landmarks.items()},
                "landmark_states": {landmark_id: landmark.state_label for landmark_id, landmark in self.landmarks.landmarks.items()},
                "dynamic_tracks": {},
                "object_state_memory": dict(self.object_state_estimates),
            }
        )

    def run(self, frames: list[FrameData]) -> SlamRunResult:
        for frame in frames:
            self._process_frame(frame)
        return SlamRunResult(
            name="pov_baseline",
            estimated_poses=self.pose_graph.estimated_poses,
            landmark_positions={k: v.position.copy() for k, v in self.landmarks.landmarks.items()},
            landmark_states={k: v.state_label for k, v in self.landmarks.landmarks.items()},
            landmark_truth={k: v.gt_object_id for k, v in self.landmarks.landmarks.items()},
            residual_history=self.residual_history,
            runtime_per_frame=self.runtime_per_frame,
            false_landmark_updates=self.landmarks.false_updates,
            frame_debug=self.frame_debug,
            object_state_estimates=dict(self.object_state_estimates),
        )
