"""Scenario construction for experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.sim.objects import DynamicMotion, MotionType, SimObject, label_radius
from src.sim.trajectory import generate_lawnmower_path
from src.sim.world import World


@dataclass
class Scenario:
    world: World
    visit_paths: list[np.ndarray]


def _sample_free_position(rng: np.random.Generator, width: float, height: float, margin: float) -> np.ndarray:
    return np.array([
        rng.uniform(margin, width - margin),
        rng.uniform(margin, height - margin),
    ], dtype=float)


def _motion_probs_for_label(label: str, motion_mix: dict[str, float]) -> np.ndarray:
    probs = np.array([
        float(motion_mix["static"]),
        float(motion_mix["semi_static"]),
        float(motion_mix["dynamic"]),
    ], dtype=float)

    # Paper-inspired semantic prior: structural objects tend to be stable,
    # while carts/cones are more plausible dynamic content in this toy world.
    if label in {"shelves", "pillars"}:
        probs *= np.array([2.4, 0.45, 0.08], dtype=float)
    elif label == "boxes":
        probs *= np.array([0.7, 2.1, 0.45], dtype=float)
    elif label == "carts":
        probs *= np.array([0.45, 0.9, 2.7], dtype=float)
    elif label == "cones":
        probs *= np.array([0.5, 0.9, 2.2], dtype=float)

    return probs / max(1e-8, probs.sum())


def build_scenario(config: dict[str, Any], rng: np.random.Generator) -> Scenario:
    world_cfg = config["world"]
    robot_cfg = config["robot"]
    width = float(world_cfg["width"])
    height = float(world_cfg["height"])
    margin = float(world_cfg.get("obstacle_margin", 1.0))
    num_visits = int(robot_cfg["num_visits"])
    steps_per_visit = int(robot_cfg["steps_per_visit"])

    motion_mix = world_cfg["motion_mix"]
    object_counts = world_cfg["object_counts"]
    patterns = list(world_cfg.get("dynamic_patterns", ["straight"]))
    motion_types = [MotionType.STATIC, MotionType.SEMI_STATIC, MotionType.DYNAMIC]

    objects: list[SimObject] = []
    object_idx = 0
    for label, count in object_counts.items():
        label_probs = _motion_probs_for_label(label, motion_mix)
        for _ in range(int(count)):
            chosen_type = motion_types[int(rng.choice(len(motion_types), p=label_probs))]
            base_pos = _sample_free_position(rng, width, height, margin)
            radius = label_radius(label)
            visit_positions: dict[int, np.ndarray] = {}
            dynamic_motion = None

            if chosen_type == MotionType.SEMI_STATIC:
                visit_positions[0] = base_pos.copy()
                for visit_idx in range(1, num_visits):
                    if rng.uniform() < 0.65:
                        shift = rng.normal(0.0, 1.4, size=2)
                        visit_positions[visit_idx] = np.clip(base_pos + shift, [margin, margin], [width - margin, height - margin])
                    else:
                        visit_positions[visit_idx] = base_pos.copy()
            elif chosen_type == MotionType.DYNAMIC:
                pattern = str(rng.choice(patterns))
                speed_scale = 0.32 if label in {"carts", "cones"} else 0.22
                velocity = rng.normal(0.0, speed_scale, size=2)
                if np.linalg.norm(velocity) < 0.12:
                    velocity += np.array([speed_scale, 0.0], dtype=float)
                waypoints = [_sample_free_position(rng, width, height, margin) for _ in range(3)]
                dynamic_motion = DynamicMotion(pattern=pattern, velocity=velocity, waypoints=waypoints)

            objects.append(
                SimObject(
                    object_id=f"obj_{object_idx:03d}",
                    label=label,
                    radius=radius,
                    motion_type=chosen_type,
                    base_position=base_pos,
                    visit_positions=visit_positions,
                    dynamic_motion=dynamic_motion,
                )
            )
            object_idx += 1

    visit_paths = [generate_lawnmower_path(width, height, steps_per_visit, margin=1.5) for _ in range(num_visits)]
    return Scenario(world=World(width=width, height=height, objects=objects), visit_paths=visit_paths)
