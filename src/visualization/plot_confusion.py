"""Confusion matrix plotting."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(confusion: dict[str, dict[str, int]], output_path: str | Path, title: str) -> None:
    labels = ["STATIC", "SEMI_STATIC", "DYNAMIC"]
    matrix = np.array([[confusion[row].get(col, 0) for col in labels] for row in labels], dtype=float)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=30)
    ax.set_yticks(range(len(labels)), labels=labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
