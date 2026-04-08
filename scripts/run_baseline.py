"""Run the baseline pipeline on a single config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.benchmark import run_suite
from src.utils.io import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    config = load_yaml(ROOT / args.config)
    outputs = run_suite(config, seed=int(config["seed"]), output_dir=config["output_dir"])
    baseline_only = outputs.metrics[outputs.metrics["run_label"] == "baseline"]
    print(baseline_only.to_string(index=False))


if __name__ == "__main__":
    main()
