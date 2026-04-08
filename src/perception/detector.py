"""Noisy object detector simulator."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.perception.observation import Observation
from src.sim.world import World
from src.utils.math_utils import rotation_matrix, wrap_angle


def simulate_observations(world: World, pose: np.ndarray, visit_idx: int, step_idx: int, config: dict[str, Any], rng: np.random.Generator) -> list[Observation]:
    robot_cfg = config["robot"]
    sensor_range = float(robot_cfg["sensor_range"])
    half_fov = math.radians(float(robot_cfg["fov_deg"])) / 2.0
    detection_noise_std = float(robot_cfg["detection_noise_std"])
    confidence_noise_std = float(robot_cfg["confidence_noise_std"])
    miss_rate = float(robot_cfg["miss_rate"])

    inv_rot = rotation_matrix(-float(pose[2]))
    observations: list[Observation] = []
    for obj in world.objects:
        position = obj.position_at(visit_idx, step_idx, world.bounds)
        delta = position - pose[:2]
        distance = float(np.linalg.norm(delta))
        bearing = wrap_angle(math.atan2(delta[1], delta[0]) - float(pose[2]))
        if distance > sensor_range or abs(bearing) > half_fov:
            continue
        if rng.uniform() < miss_rate:
            continue
        noisy_world = position + rng.normal(0.0, detection_noise_std, size=2)
        relative = inv_rot @ (noisy_world - pose[:2])
        confidence = float(np.clip(1.0 - distance / sensor_range + rng.normal(0.0, confidence_noise_std), 0.05, 1.0))
        observations.append(
            Observation(
                object_id=obj.object_id,
                label=obj.label,
                world_position=noisy_world,
                relative_position=relative,
                confidence=confidence,
                radius=obj.radius,
                visit_idx=visit_idx,
                step_idx=step_idx,
            )
        )
    return observations
