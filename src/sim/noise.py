"""Noise helpers."""

from __future__ import annotations

import numpy as np


def gaussian_2d(std: float, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0.0, std, size=2)
