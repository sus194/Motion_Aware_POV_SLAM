"""Plot experiment outputs from saved metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.visualization.animate_runs import animate_run
from src.visualization.plot_confusion import plot_confusion_matrix
from src.visualization.plot_maps import plot_map_comparison
from src.visualization.plot_trajectories import plot_metric_bars, plot_stress_curve


def _select_run_file(runs_dir: Path, method: str) -> Path | None:
    candidates = sorted(runs_dir.glob(f"*_{method}_seed*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
    fallback = sorted(runs_dir.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    return fallback[0] if fallback else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", default="outputs/metrics")
    parser.add_argument("--method", default="full_motion_aware", help="Run label to use for map/confusion/animation outputs")
    args = parser.parse_args()

    metrics_path = ROOT / args.metrics_dir / "aggregate_results.csv"
    metrics = pd.read_csv(metrics_path)
    figures_dir = ROOT / "outputs" / "figures"

    plot_metric_bars(metrics, "ate", figures_dir / "ate_comparison.png", "ATE Comparison")
    plot_metric_bars(metrics, "state_classification_accuracy", figures_dir / "classification_accuracy.png", "State Classification Accuracy")

    dynamic_field = "world.motion_mix.dynamic"
    if dynamic_field in metrics.columns:
        stress_dynamic = metrics.dropna(subset=[dynamic_field])
        if not stress_dynamic.empty:
            plot_stress_curve(stress_dynamic, dynamic_field, "ate", figures_dir / "stress_dynamic_fraction.png", "ATE vs Dynamic Fraction")

    noise_field = "robot.detection_noise_std"
    if noise_field in metrics.columns:
        stress_noise = metrics.dropna(subset=[noise_field])
        if not stress_noise.empty:
            plot_stress_curve(stress_noise, noise_field, "ate", figures_dir / "stress_noise.png", "ATE vs Detection Noise")

    run_file = _select_run_file(ROOT / "outputs" / "runs", args.method)
    if run_file:
        with run_file.open("r", encoding="utf-8") as handle:
            run_summary = json.load(handle)
        output_prefix = args.method
        plot_map_comparison(run_summary, figures_dir / f"{output_prefix}_map.png")
        confusion = run_summary["metrics"].get("confusion_matrix")
        if confusion:
            plot_confusion_matrix(confusion, figures_dir / f"{output_prefix}_confusion.png", f"Motion-State Confusion: {args.method}")
        if run_summary.get("frame_truth") and run_summary.get("frame_debug"):
            animate_run(run_summary, figures_dir / f"{output_prefix}_animation.gif")
        print(f"Used run file for qualitative plots: {run_file}")

    print(f"Saved figures to {figures_dir}")


if __name__ == "__main__":
    main()
