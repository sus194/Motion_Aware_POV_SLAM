"""Motion-aware extension of the baseline object-aware SLAM pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.perception.data_association import associate_position
from src.perception.observation import FrameData, Observation
from src.slam.baseline_slam import SlamRunResult
from src.slam.landmark_manager import LandmarkManager
from src.slam.optimizer import weighted_pose_correction
from src.slam.pose_graph import PoseGraph
from src.slam.state_classifier import MotionStateClassifier
from src.tracking.kalman_tracker import ConstantVelocityKalman
from src.tracking.motion_estimator import MotionFeatureBank
from src.tracking.track_manager import TrackManager
from src.utils.math_utils import local_to_world, pose_add


@dataclass
class MotionAwareComponents:
    landmarks: LandmarkManager
    tracks: TrackManager
    features: MotionFeatureBank
    classifier: MotionStateClassifier


class MotionAwareSLAM:
    """Separate-map dynamic SLAM approximation inspired by dynamic-content SLAM papers."""

    def __init__(self, config: dict[str, Any], mode: str = "full") -> None:
        slam_cfg = config["slam"]
        self.mode = mode
        self.pose_graph = PoseGraph()
        self.current_pose = np.zeros(3, dtype=float)
        self.association_distance = float(slam_cfg["association_distance"])
        self.pose_correction_gain = float(slam_cfg["pose_correction_gain"])
        self.relocation_confirmations = int(slam_cfg["relocation_confirmations"])
        self.relocation_radius = float(slam_cfg["relocation_radius"])
        self.dynamic_prior_labels = set(slam_cfg.get("dynamic_prior_labels", []))
        self.weight_lookup = {
            "STATIC": float(slam_cfg["static_weight"]),
            "SEMI_STATIC": float(slam_cfg["semistatic_weight"]),
            "DYNAMIC": float(slam_cfg["dynamic_weight"]),
        }
        tracker = ConstantVelocityKalman(process_var=float(slam_cfg["tracker_process_var"]), measurement_var=float(slam_cfg["tracker_measurement_var"]))
        self.components = MotionAwareComponents(
            landmarks=LandmarkManager(update_alpha=float(slam_cfg["landmark_update_alpha"])),
            tracks=TrackManager(tracker=tracker, stale_frames=int(slam_cfg["tracker_stale_frames"])),
            features=MotionFeatureBank(),
            classifier=MotionStateClassifier(
                dynamic_velocity_threshold=float(slam_cfg["dynamic_velocity_threshold"]),
                semi_static_revisit_threshold=float(slam_cfg["semi_static_revisit_threshold"]),
                persistence_threshold=int(slam_cfg["persistence_threshold"]),
            ),
        )
        self.residual_history: list[float] = []
        self.runtime_per_frame: list[float] = []
        self.dynamic_track_errors: list[float] = []
        self.frame_debug: list[dict[str, Any]] = []
        self.object_state_memory: dict[str, str] = {}
        self.initialized = False
        self.current_visit: int | None = None
        self.visit_initial_noise_xy = float(slam_cfg.get("visit_initial_noise_xy", 0.0))
        self.visit_initial_noise_theta = float(slam_cfg.get("visit_initial_noise_theta", 0.0))
        self.stable_landmark_min_observations = int(slam_cfg.get("stable_landmark_min_observations", 3))
        self.stable_landmark_max_residual = float(slam_cfg.get("stable_landmark_max_residual", 1.2))

    def _initial_pose_for_visit(self, frame: FrameData) -> np.ndarray:
        pose = frame.true_pose.copy()
        if frame.visit_idx > 0:
            sign = -1.0 if frame.visit_idx % 2 else 1.0
            pose[0] += sign * self.visit_initial_noise_xy
            pose[1] -= 0.6 * self.visit_initial_noise_xy
            pose[2] += sign * self.visit_initial_noise_theta
        return pose

    def _state_weight(self, label: str) -> float:
        if self.mode == "filter_only" and label == "DYNAMIC":
            return self.weight_lookup["dynamic"]
        if self.mode == "tracking_only":
            return self.weight_lookup["STATIC"]
        return self.weight_lookup.get(label, 1.0)

    def _classify_observation(self, obs: Observation, features) -> str:
        dynamic_threshold = self.components.classifier.dynamic_velocity_threshold
        strong_motion_threshold = 2.2 * dynamic_threshold
        if features.observation_count >= self.components.classifier.persistence_threshold and features.revisit_jump >= self.components.classifier.semi_static_revisit_threshold:
            state_label = "SEMI_STATIC"
        else:
            state_label = "STATIC"

        has_dynamic_prior = obs.label in self.dynamic_prior_labels
        prior_dynamic = (
            has_dynamic_prior
            and features.observation_count >= 3
            and features.avg_displacement >= dynamic_threshold
            and features.avg_residual >= 0.35
        )
        geometry_dynamic = (
            features.observation_count >= 3
            and has_dynamic_prior
            and features.avg_displacement >= strong_motion_threshold
            and features.avg_residual >= 0.7
        )
        persistent_dynamic = (
            self.object_state_memory.get(obs.object_id) == "DYNAMIC"
            and has_dynamic_prior
            and features.avg_displacement >= dynamic_threshold
            and features.avg_residual >= 0.35
        )
        if prior_dynamic or geometry_dynamic or persistent_dynamic:
            state_label = "DYNAMIC"

        if self.mode in {"filter_only", "tracking_only"}:
            state_label = "DYNAMIC" if prior_dynamic or geometry_dynamic else "STATIC"
        self.object_state_memory[obs.object_id] = state_label
        return state_label
    def _snapshot_frame(self, frame: FrameData) -> None:
        self.frame_debug.append(
            {
                "visit_idx": frame.visit_idx,
                "step_idx": frame.step_idx,
                "estimated_pose": self.current_pose.copy().tolist(),
                "landmarks": {landmark_id: landmark.position.copy().tolist() for landmark_id, landmark in self.components.landmarks.landmarks.items()},
                "landmark_states": {landmark_id: landmark.state_label for landmark_id, landmark in self.components.landmarks.landmarks.items()},
                "dynamic_tracks": {
                    track_id: {
                        "position": track.state[:2].copy().tolist(),
                        "velocity": track.state[2:].copy().tolist(),
                        "label": track.label,
                        "object_id": track.object_id,
                    }
                    for track_id, track in self.components.tracks.tracks.items()
                },
                "object_state_memory": dict(self.object_state_memory),
                "correction_support": getattr(self, "last_correction_support", 0),
            }
        )

    def _update_dynamic_track(self, obs: Observation, measured_world: np.ndarray) -> None:
        track_match = associate_position(obs.label, measured_world, self.components.tracks.active_candidates(), self.association_distance)
        tracked_obs = Observation(
            object_id=obs.object_id,
            label=obs.label,
            world_position=measured_world,
            relative_position=obs.relative_position,
            confidence=obs.confidence,
            radius=obs.radius,
            visit_idx=obs.visit_idx,
            step_idx=obs.step_idx,
        )
        if track_match.match_id is None:
            self.components.tracks.spawn_track(tracked_obs)
        else:
            self.components.tracks.update_track(track_match.match_id, tracked_obs)
            track = self.components.tracks.tracks[track_match.match_id]
            self.dynamic_track_errors.append(float(np.linalg.norm(track.state[:2] - measured_world)))

    def _process_frame(self, frame: FrameData) -> None:
        if (not self.initialized) or (self.current_visit != frame.visit_idx):
            self.current_pose = self._initial_pose_for_visit(frame)
            self.initialized = True
            self.current_visit = frame.visit_idx
        else:
            self.current_pose = pose_add(self.current_pose, frame.odom_delta)
        self.components.tracks.predict_all()
        residuals: list[np.ndarray] = []
        weights: list[float] = []

        for obs in frame.observations:
            measured_world = local_to_world(self.current_pose, obs.relative_position)
            landmark_match = associate_position(obs.label, measured_world, self.components.landmarks.candidates(), self.association_distance)
            landmark_residual = landmark_match.distance if landmark_match.match_id is not None else self.association_distance
            features = self.components.features.update(obs.object_id, frame.visit_idx, measured_world, landmark_residual)
            state_label = self._classify_observation(obs, features)

            if state_label == "DYNAMIC":
                self._update_dynamic_track(obs, measured_world)
                continue

            if landmark_match.match_id is None:
                landmark_id = self.components.landmarks.add(obs.label, measured_world, gt_object_id=obs.object_id)
                self.components.landmarks.landmarks[landmark_id].state_label = state_label
                continue

            landmark = self.components.landmarks.landmarks[landmark_match.match_id]
            landmark_residual_vec = measured_world - landmark.position
            landmark_residual = float(np.linalg.norm(landmark_residual_vec))
            self.residual_history.append(landmark_residual)
            landmark.state_label = state_label

            correction_anchor = landmark.position.copy()
            pose_state_label = "DYNAMIC" if state_label == "DYNAMIC" else "STATIC"
            weight = self._state_weight(pose_state_label) * obs.confidence
            if weight > 1e-6:
                residuals.append(correction_anchor - measured_world)
                weights.append(weight)

            if state_label == "SEMI_STATIC" and landmark_residual > self.relocation_radius:
                self.components.landmarks.consider_relocation(landmark_match.match_id, measured_world, self.relocation_radius, self.relocation_confirmations, obs.object_id)
            else:
                self.components.landmarks.update_position(landmark_match.match_id, measured_world, landmark_residual)

        self.last_correction_support = len(residuals)
        correction = weighted_pose_correction(residuals, weights, self.pose_correction_gain)
        self.current_pose[:2] += correction
        self.pose_graph.add_pose(self.current_pose, correction)
        self.components.tracks.prune()
        self._snapshot_frame(frame)

    def run(self, frames: list[FrameData]) -> SlamRunResult:
        for frame in frames:
            self._process_frame(frame)
        return SlamRunResult(
            name=f"motion_aware_{self.mode}",
            estimated_poses=self.pose_graph.estimated_poses,
            landmark_positions={k: v.position.copy() for k, v in self.components.landmarks.landmarks.items()},
            landmark_states={k: v.state_label for k, v in self.components.landmarks.landmarks.items()},
            landmark_truth={k: v.gt_object_id for k, v in self.components.landmarks.landmarks.items()},
            residual_history=self.residual_history,
            runtime_per_frame=self.runtime_per_frame,
            false_landmark_updates=self.components.landmarks.false_updates,
            track_stats={
                "dynamic_track_position_error": float(np.mean(self.dynamic_track_errors)) if self.dynamic_track_errors else 0.0,
                "track_survival_length": float(np.mean([track.age for track in self.components.tracks.tracks.values()])) if self.components.tracks.tracks else 0.0,
                "id_switch_count": float(self.components.tracks.id_switches),
            },
            frame_debug=self.frame_debug,
            object_state_estimates=dict(self.object_state_memory),
        )














