"""Weighted pose correction utilities."""

from __future__ import annotations

import numpy as np


def weighted_pose_correction(residuals: list[np.ndarray], weights: list[float], gain: float) -> np.ndarray:
    if not residuals:
        return np.zeros(2, dtype=float)
    residual_stack = np.stack(residuals, axis=0)
    weight_array = np.array(weights, dtype=float).reshape(-1, 1)
    weighted = (residual_stack * weight_array).sum(axis=0) / max(1e-6, weight_array.sum())
    return gain * weighted


def robust_translation_correction(
    residuals: list[np.ndarray],
    weights: list[float],
    gain: float,
    min_support: int,
    trim_quantile: float,
    max_step: float,
) -> np.ndarray:
    """Estimate a robust 2D translation correction from multiple landmark residuals."""
    if len(residuals) < min_support:
        return np.zeros(2, dtype=float)

    residual_stack = np.stack(residuals, axis=0)
    norms = np.linalg.norm(residual_stack, axis=1)
    cutoff = float(np.quantile(norms, trim_quantile)) if len(norms) > min_support else float(np.max(norms))
    keep = norms <= max(cutoff, 1e-9)
    if int(np.sum(keep)) < min_support:
        return np.zeros(2, dtype=float)

    kept_residuals = residual_stack[keep]
    kept_weights = np.array(weights, dtype=float)[keep]
    weight_sum = float(np.sum(kept_weights))
    if weight_sum <= 1e-9:
        return np.zeros(2, dtype=float)

    median = np.median(kept_residuals, axis=0)
    weighted_mean = (kept_residuals * kept_weights.reshape(-1, 1)).sum(axis=0) / weight_sum
    correction = gain * (0.6 * median + 0.4 * weighted_mean)

    step = float(np.linalg.norm(correction))
    if step > max_step:
        correction = correction * (max_step / step)
    return correction
