"""Top-down map plotting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def plot_map_comparison(run_summary: dict[str, Any], output_path: str | Path) -> None:
    true_traj = np.array(run_summary["trajectory_true"], dtype=float)
    est_traj = np.array(run_summary["trajectory_estimated"], dtype=float)
    landmarks = run_summary["landmarks"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(true_traj[:, 0], true_traj[:, 1], label="GT trajectory", linewidth=2)
    ax.plot(est_traj[:, 0], est_traj[:, 1], label="Estimated trajectory", linestyle="--", linewidth=2)
    if landmarks:
        landmark_points = np.stack([np.array(v, dtype=float) for v in landmarks.values()], axis=0)
        ax.scatter(landmark_points[:, 0], landmark_points[:, 1], c="tab:red", label="Estimated landmarks", alpha=0.7)
    ax.set_title(f"Map View: {run_summary['method']}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(True, alpha=0.3)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
