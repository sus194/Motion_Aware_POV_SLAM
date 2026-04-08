"""Run experiment suites and save aggregate results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.ablations import summarize_ablations
from src.evaluation.benchmark import run_suite
from src.utils.io import deep_update, load_yaml, save_json, set_nested


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/experiments.yaml")
    args = parser.parse_args()

    suite_cfg = load_yaml(ROOT / args.config)
    base_config = load_yaml(ROOT / suite_cfg["base_config"])
    rows = []

    for scenario in suite_cfg["scenarios"]:
        scenario_name = scenario["name"]
        seed_list = suite_cfg.get("seed_list", [base_config["seed"]])
        grid = scenario.get("grid")
        grid_values = grid["values"] if grid else [None]
        for grid_value in grid_values:
            for seed in seed_list:
                config = deep_update(base_config, scenario.get("overrides", {}))
                config["experiment_name"] = f"{scenario_name}_{seed}" if grid_value is None else f"{scenario_name}_{grid_value}_{seed}"
                if grid:
                    set_nested(config, grid["field"], grid_value)
                outputs = run_suite(config, seed=seed, output_dir=config["output_dir"])
                df = outputs.metrics.copy()
                df["scenario"] = scenario_name
                if grid:
                    df[grid["field"]] = grid_value
                rows.append(df)

    aggregate = pd.concat(rows, ignore_index=True)
    aggregate.to_csv(ROOT / "outputs" / "metrics" / "aggregate_results.csv", index=False)
    summarize_ablations(aggregate).to_csv(ROOT / "outputs" / "metrics" / "ablation_summary.csv", index=False)
    save_json(ROOT / "outputs" / "metrics" / "aggregate_preview.json", aggregate.head(20).to_dict(orient="records"))
    print(aggregate.groupby(["scenario", "run_label"])["ate"].mean().reset_index().to_string(index=False))


if __name__ == "__main__":
    main()
