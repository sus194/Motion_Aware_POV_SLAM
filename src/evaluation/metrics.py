"""Metrics for localization, mapping, classification, and tracking."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from src.sim.objects import MotionType
from src.slam.baseline_slam import SlamRunResult


def absolute_trajectory_error(true_poses: np.ndarray, estimated_poses: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(true_poses[:, :2] - estimated_poses[:, :2], axis=1)))


def relative_pose_error(true_poses: np.ndarray, estimated_poses: np.ndarray) -> float:
    if len(true_poses) < 2:
        return 0.0
    true_delta = np.diff(true_poses[:, :2], axis=0)
    est_delta = np.diff(estimated_poses[:, :2], axis=0)
    return float(np.mean(np.linalg.norm(true_delta - est_delta, axis=1)))


def map_metrics(run: SlamRunResult, gt_positions: dict[str, np.ndarray]) -> dict[str, float]:
    errors = []
    matched = 0
    consistency = []
    for landmark_id, gt_object_id in run.landmark_truth.items():
        if gt_object_id is None or gt_object_id not in gt_positions:
            continue
        matched += 1
        error = float(np.linalg.norm(run.landmark_positions[landmark_id] - gt_positions[gt_object_id]))
        errors.append(error)
        consistency.append(max(0.0, 1.0 - error / 5.0))
    return {
        "landmark_position_error": float(np.mean(errors)) if errors else 0.0,
        "map_consistency_score": float(np.mean(consistency)) if consistency else 0.0,
        "false_landmark_updates": float(run.false_landmark_updates),
        "matched_landmarks": float(matched),
    }


def _object_level_predictions(run: SlamRunResult) -> dict[str, str]:
    predictions = dict(run.object_state_estimates)
    if predictions:
        return predictions
    for landmark_id, gt_object_id in run.landmark_truth.items():
        if gt_object_id is not None:
            predictions[gt_object_id] = run.landmark_states.get(landmark_id, "STATIC")
    return predictions


def classification_metrics(run: SlamRunResult, gt_motion: dict[str, MotionType]) -> dict[str, Any]:
    truth: list[str] = []
    pred: list[str] = []
    labels = ["STATIC", "SEMI_STATIC", "DYNAMIC"]
    confusion = {label: Counter({inner: 0 for inner in labels}) for label in labels}
    predictions = _object_level_predictions(run)
    for object_id, motion_type in gt_motion.items():
        gt_label = motion_type.value
        pred_label = predictions.get(object_id, "STATIC")
        truth.append(gt_label)
        pred.append(pred_label)
        confusion[gt_label][pred_label] += 1
    accuracy = float(sum(t == p for t, p in zip(truth, pred)) / len(truth)) if truth else 0.0
    return {
        "state_classification_accuracy": accuracy,
        "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
    }


def change_detection_metrics(run: SlamRunResult, gt_motion: dict[str, MotionType]) -> dict[str, float]:
    tp = fp = fn = 0
    predictions = _object_level_predictions(run)
    for object_id, motion_type in gt_motion.items():
        gt_changed = motion_type != MotionType.STATIC
        pred_changed = predictions.get(object_id, "STATIC") != "STATIC"
        tp += int(gt_changed and pred_changed)
        fp += int((not gt_changed) and pred_changed)
        fn += int(gt_changed and (not pred_changed))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "change_precision": float(precision),
        "change_recall": float(recall),
    }
