"""Interactive run viewer for saved playback JSON files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.visualization.animate_runs import show_run_viewer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Path to a saved outputs/runs/*.json file")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--interval-ms", type=int, default=100)
    parser.add_argument("--step", action="store_true", help="Open paused for manual frame-by-frame stepping")
    args = parser.parse_args()

    run_path = ROOT / args.run if not Path(args.run).is_absolute() else Path(args.run)
    with run_path.open("r", encoding="utf-8") as handle:
        run_summary = json.load(handle)

    if not run_summary.get("frame_truth") or not run_summary.get("frame_debug"):
        raise ValueError("This run file does not contain playback data. Re-run the experiment with the updated code.")

    show_run_viewer(
        run_summary,
        start_frame=max(0, args.start_frame),
        autoplay=not args.step,
        interval_ms=max(10, args.interval_ms),
    )


if __name__ == "__main__":
    main()
