"""Experiment harness for baseline and motion-aware comparisons."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.metrics import absolute_trajectory_error, change_detection_metrics, classification_metrics, map_metrics, relative_pose_error
from src.perception.detector import simulate_observations
from src.perception.observation import FrameData
from src.sim.objects import MotionType, object_to_dict
from src.sim.robot import RobotConfig, simulate_odometry
from src.sim.scenario_builder import build_scenario
from src.slam.baseline_slam import BaselineObjectSLAM, SlamRunResult
from src.slam.motion_aware_slam import MotionAwareSLAM
from src.utils.io import ensure_dirs, save_json
from src.utils.random_seed import set_global_seed


@dataclass
class BenchmarkOutputs:
    metrics: pd.DataFrame
    run_summaries: list[dict[str, Any]]


def _simulate_frames(config: dict[str, Any], seed: int) -> tuple[list[FrameData], np.ndarray, dict[str, np.ndarray], dict[str, MotionType], dict[str, Any]]:
    rng = set_global_seed(seed)
    scenario = build_scenario(config, rng)
    robot_cfg = config["robot"]
    frames: list[FrameData] = []
    true_poses = []
    gt_positions: dict[str, np.ndarray] = {}
    gt_motion: dict[str, MotionType] = {}
    frame_truth: list[dict[str, Any]] = []

    odom_cfg = RobotConfig(
        sensor_range=float(robot_cfg["sensor_range"]),
        fov_deg=float(robot_cfg["fov_deg"]),
        odom_noise_xy=float(robot_cfg["odom_noise_std"]["xy"]),
        odom_noise_theta=float(robot_cfg["odom_noise_std"]["theta"]),
    )

    for visit_idx, gt_path in enumerate(scenario.visit_paths):
        odom = simulate_odometry(gt_path, odom_cfg, rng)
        visit_deltas = [np.zeros(3, dtype=float)] + [odom[idx] for idx in range(len(odom))]
        for step_idx, (pose, delta) in enumerate(zip(gt_path, visit_deltas)):
            observations = simulate_observations(scenario.world, pose, visit_idx, step_idx, config, rng)
            frames.append(FrameData(visit_idx=visit_idx, step_idx=step_idx, true_pose=pose, odom_delta=delta, observations=observations))
            true_poses.append(pose)
            frame_truth.append(
                {
                    "visit_idx": visit_idx,
                    "step_idx": step_idx,
                    "true_pose": pose.tolist(),
                    "objects": {
                        obj.object_id: {
                            "label": obj.label,
                            "motion_type": obj.motion_type.value,
                            "position": obj.position_at(visit_idx, step_idx, scenario.world.bounds).tolist(),
                            "radius": obj.radius,
                        }
                        for obj in scenario.world.objects
                    },
                    "observations": [
                        {
                            "object_id": obs.object_id,
                            "label": obs.label,
                            "world_position": obs.world_position.tolist(),
                            "confidence": obs.confidence,
                        }
                        for obs in observations
                    ],
                }
            )

    final_visit = int(config["robot"]["num_visits"]) - 1
    final_step = int(config["robot"]["steps_per_visit"]) - 1
    for obj in scenario.world.objects:
        gt_positions[obj.object_id] = obj.position_at(final_visit, final_step, scenario.world.bounds)
        gt_motion[obj.object_id] = obj.motion_type

    summary = {
        "world": {
            "width": scenario.world.width,
            "height": scenario.world.height,
            "objects": [object_to_dict(obj) for obj in scenario.world.objects],
        },
        "frame_truth": frame_truth,
    }
    return frames, np.stack(true_poses, axis=0), gt_positions, gt_motion, summary


def _evaluate_run(run: SlamRunResult, true_poses: np.ndarray, gt_positions: dict[str, np.ndarray], gt_motion: dict[str, MotionType]) -> dict[str, Any]:
    estimated_poses = np.stack(run.estimated_poses, axis=0) if run.estimated_poses else np.zeros_like(true_poses)
    metrics: dict[str, Any] = {
        "method": run.name,
        "ate": absolute_trajectory_error(true_poses, estimated_poses),
        "rpe": relative_pose_error(true_poses, estimated_poses),
        "avg_runtime_per_frame": float(np.mean(run.runtime_per_frame)) if run.runtime_per_frame else 0.0,
        "total_runtime": float(np.sum(run.runtime_per_frame)) if run.runtime_per_frame else 0.0,
    }
    metrics.update(map_metrics(run, gt_positions))
    metrics.update(classification_metrics(run, gt_motion))
    metrics.update(change_detection_metrics(run, gt_motion))
    metrics.update(run.track_stats)
    return metrics


def run_suite(config: dict[str, Any], seed: int, output_dir: str) -> BenchmarkOutputs:
    ensure_dirs(Path(output_dir) / "figures", Path(output_dir) / "metrics", Path(output_dir) / "logs", Path(output_dir) / "runs")
    frames, true_poses, gt_positions, gt_motion, summary = _simulate_frames(config, seed)

    runners = [
        ("baseline", BaselineObjectSLAM(config)),
        ("dynamic_filter_only", MotionAwareSLAM(config, mode="filter_only")),
        ("kalman_tracking_only", MotionAwareSLAM(config, mode="tracking_only")),
        ("full_motion_aware", MotionAwareSLAM(config, mode="full")),
    ]

    rows = []
    run_summaries = []
    for label, runner in runners:
        started = time.perf_counter()
        run = runner.run(frames)
        elapsed = time.perf_counter() - started
        if not run.runtime_per_frame:
            run.runtime_per_frame = [elapsed / max(1, len(frames))] * len(frames)
        metrics = _evaluate_run(run, true_poses, gt_positions, gt_motion)
        metrics["seed"] = seed
        metrics["run_label"] = label
        rows.append(metrics)

        run_summary = {
            "seed": seed,
            "method": label,
            "metrics": metrics,
            "trajectory_true": true_poses.tolist(),
            "trajectory_estimated": np.stack(run.estimated_poses, axis=0).tolist(),
            "landmarks": {k: v.tolist() for k, v in run.landmark_positions.items()},
            "landmark_states": run.landmark_states,
            "frame_debug": run.frame_debug,
        }
        run_summaries.append(run_summary)
        save_json(Path(output_dir) / "runs" / f"{config['experiment_name']}_{label}_seed{seed}.json", {**summary, **run_summary})

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(Path(output_dir) / "metrics" / f"{config['experiment_name']}_seed{seed}.csv", index=False)
    return BenchmarkOutputs(metrics=metrics_df, run_summaries=run_summaries)
