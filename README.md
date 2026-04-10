# Motion-Aware POV-SLAM

A simulation-only research prototype for **object-level SLAM in semi-static and dynamic indoor environments**.

This repository studies a lightweight **POV-SLAM-style baseline** and extends it with a **motion-aware layer** that separates objects into three states:

- **STATIC** — reliable long-term landmarks
- **SEMI_STATIC** — objects that may move between revisits but remain useful after confirmation
- **DYNAMIC** — continuously moving objects that should be tracked separately instead of fused into the stationary map

The project is intentionally small and reproducible: it uses a 2D simulator, config-driven experiments, saved run summaries, aggregate metrics, plotting utilities, and a LaTeX report.

---

## What this repo includes

- A **2D world simulator** with static, semi-static, and dynamic object categories
- A **POV-style baseline** for object-aware SLAM under map change
- A **motion-aware SLAM variant** with:
  - heuristic motion-state classification
  - dynamic-object filtering
  - sidecar constant-velocity Kalman tracking
  - semi-static relocation logic
- A **benchmark harness** for running multi-seed experiment suites
- Plotting and playback tools for qualitative and quantitative inspection
- A LaTeX report documenting the motivation, method, and results

---

## Method summary

The repository compares four modes:

1. **baseline** — simplified POV-SLAM-style object consistency reasoning
2. **dynamic_filter_only** — excludes dynamic objects from stationary landmark correction
3. **kalman_tracking_only** — adds sidecar dynamic tracking
4. **full_motion_aware** — combines motion-state classification, dynamic filtering, tracking, and semi-static update logic

At a high level, the motion-aware pipeline works like this:

1. simulate a world with multiple visits and object motion patterns
2. generate noisy odometry and object detections
3. classify object behavior as static, semi-static, or dynamic
4. use static/semi-static objects for map updates and pose correction
5. route dynamic objects to a separate tracker instead of treating them as fixed landmarks
6. save metrics, run summaries, and figures for later analysis

---

## Repository layout

```text
Motion_Aware_POV_SLAM/
├── config/
│   ├── default.yaml            # single-run default configuration
│   └── experiments.yaml        # benchmark suite definition
├── scripts/
│   ├── run_baseline.py         # run the baseline on one config
│   ├── run_motion_aware.py     # run the full motion-aware pipeline on one config
│   ├── run_experiments.py      # run multi-scenario, multi-seed benchmark suite
│   ├── plot_results.py         # generate figures from saved metrics and runs
│   └── view_run.py             # interactive playback for a saved run JSON
├── src/
│   ├── evaluation/             # benchmark harness and metrics
│   ├── perception/             # simulated detections and association
│   ├── sim/                    # world, robot, trajectory, scenario generation
│   ├── slam/                   # baseline and motion-aware SLAM logic
│   ├── tracking/               # motion features and Kalman-based tracking
│   ├── utils/                  # YAML/JSON IO, math, logging, seeding
│   └── visualization/          # plots, confusion matrix, animation tools
├── outputs/
│   ├── metrics/                # CSV/JSON metric outputs
│   ├── runs/                   # per-run playback/debug summaries
│   ├── figures/                # generated plots and animations
│   └── logs/                   # optional logs
├── report/
│   ├── main.tex                # LaTeX report
│   └── README.md               # report build notes
├── requirements.txt
└── README.md
