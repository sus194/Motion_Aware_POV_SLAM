"""Trajectory and benchmark comparison plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_metric_bars(metrics: pd.DataFrame, metric: str, output_path: str | Path, title: str) -> None:
    grouped = metrics.groupby("run_label")[metric].mean().sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    grouped.plot(kind="bar", ax=ax, color=["#5b8ff9", "#61ddaa", "#65789b", "#f6bd16"])
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.grid(True, axis="y", alpha=0.3)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_stress_curve(metrics: pd.DataFrame, x_field: str, y_field: str, output_path: str | Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for method, group in metrics.groupby("run_label"):
        series = group.groupby(x_field)[y_field].mean().sort_index()
        ax.plot(series.index, series.values, marker="o", label=method)
    ax.set_title(title)
    ax.set_xlabel(x_field)
    ax.set_ylabel(y_field)
    ax.grid(True, alpha=0.3)
    ax.legend()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
