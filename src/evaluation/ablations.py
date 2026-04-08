"""Ablation helpers."""

from __future__ import annotations

import pandas as pd


def summarize_ablations(metrics: pd.DataFrame) -> pd.DataFrame:
    value_columns = [col for col in metrics.columns if col not in {"seed", "method", "run_label", "confusion_matrix"}]
    return metrics.groupby("run_label")[value_columns].mean(numeric_only=True).reset_index()
