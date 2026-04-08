"""Random seeding helpers."""

from __future__ import annotations

import random

import numpy as np


def set_global_seed(seed: int) -> np.random.Generator:
    """Seed Python and NumPy RNGs and return a generator."""
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)
