"""Top-down playback and animation utilities for simulated runs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np


TRUTH_COLORS = {
    "STATIC": "#4C78A8",
    "SEMI_STATIC": "#F58518",
    "DYNAMIC": "#E45756",
}

STATE_COLORS = {
    "STATIC": "#54A24B",
    "SEMI_STATIC": "#EECA3B",
    "DYNAMIC": "#B279A2",
}


def _bounds(run_summary: dict) -> tuple[float, float]:
    world = run_summary.get("world", {})
    return float(world.get("width", 30.0)), float(world.get("height", 20.0))


def _draw_frame(ax: plt.Axes, run_summary: dict, frame_idx: int) -> None:
    width, height = _bounds(run_summary)
    truth_frame = run_summary["frame_truth"][frame_idx]
    debug_frame = run_summary["frame_debug"][frame_idx]
    true_traj = np.array(run_summary["trajectory_true"], dtype=float)
    est_traj = np.array(run_summary["trajectory_estimated"], dtype=float)

    ax.clear()
    ax.set_xlim(0.0, width)
    ax.set_ylim(0.0, height)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{run_summary['method']} | visit {truth_frame['visit_idx']} step {truth_frame['step_idx']}")

    ax.plot(true_traj[: frame_idx + 1, 0], true_traj[: frame_idx + 1, 1], color="#1f77b4", linewidth=2, label="GT path")
    ax.plot(est_traj[: frame_idx + 1, 0], est_traj[: frame_idx + 1, 1], color="#ff7f0e", linewidth=2, linestyle="--", label="Estimated path")

    true_pose = np.array(truth_frame["true_pose"], dtype=float)
    est_pose = np.array(debug_frame["estimated_pose"], dtype=float)
    ax.scatter(true_pose[0], true_pose[1], color="#1f77b4", s=80, marker="o", zorder=5, label="GT robot")
    ax.scatter(est_pose[0], est_pose[1], color="#ff7f0e", s=90, marker="x", zorder=5, label="Estimated robot")

    for obj in truth_frame["objects"].values():
        position = np.array(obj["position"], dtype=float)
        motion_type = obj["motion_type"]
        radius = float(obj.get("radius", 0.4))
        patch = Circle(position, radius=radius, facecolor=TRUTH_COLORS.get(motion_type, "#999999"), edgecolor="black", alpha=0.25)
        ax.add_patch(patch)
        ax.text(position[0], position[1] + radius + 0.12, f"{obj['label']}\n{motion_type}", fontsize=7, ha="center", va="bottom")

    for landmark_id, position in debug_frame.get("landmarks", {}).items():
        pos = np.array(position, dtype=float)
        state = debug_frame.get("landmark_states", {}).get(landmark_id, "STATIC")
        ax.scatter(pos[0], pos[1], color=STATE_COLORS.get(state, "#333333"), s=45, marker="s", zorder=6)
        ax.text(pos[0], pos[1] - 0.18, state, fontsize=6, ha="center", va="top", color=STATE_COLORS.get(state, "#333333"))

    for track_id, track in debug_frame.get("dynamic_tracks", {}).items():
        pos = np.array(track["position"], dtype=float)
        vel = np.array(track["velocity"], dtype=float)
        ax.scatter(pos[0], pos[1], color="#D81B60", s=55, marker="D", zorder=7)
        ax.arrow(pos[0], pos[1], vel[0], vel[1], width=0.02, head_width=0.18, length_includes_head=True, color="#D81B60", alpha=0.8)
        ax.text(pos[0], pos[1] + 0.22, track_id, fontsize=6, ha="center", va="bottom", color="#D81B60")

    observation_points = [obs["world_position"] for obs in truth_frame.get("observations", [])]
    if observation_points:
        obs = np.array(observation_points, dtype=float)
        ax.scatter(obs[:, 0], obs[:, 1], color="#6C757D", s=18, alpha=0.4, marker=".", label="detections")

    handles, labels = ax.get_legend_handles_labels()
    dedup: dict[str, object] = {}
    for handle, label in zip(handles, labels):
        dedup.setdefault(label, handle)
    ax.legend(dedup.values(), dedup.keys(), loc="upper right", fontsize=8)


def animate_run(run_summary: dict, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    def update(frame_idx: int):
        _draw_frame(ax, run_summary, frame_idx)
        return []

    ani = animation.FuncAnimation(fig, update, frames=len(run_summary["frame_truth"]), interval=100, blit=False)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        ani.save(output_path, dpi=140)
    except Exception:
        fallback = Path(output_path).with_suffix(".png")
        _draw_frame(ax, run_summary, len(run_summary["frame_truth"]) - 1)
        fig.savefig(fallback, dpi=140)
    plt.close(fig)


def show_run_viewer(run_summary: dict, start_frame: int = 0, autoplay: bool = True, interval_ms: int = 100) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    total_frames = len(run_summary["frame_truth"])
    state = {"frame_idx": max(0, min(start_frame, total_frames - 1)), "paused": not autoplay}
    _draw_frame(ax, run_summary, state["frame_idx"])

    def draw_current() -> None:
        _draw_frame(ax, run_summary, state["frame_idx"])
        fig.canvas.draw_idle()

    def update(_: int):
        if not state["paused"]:
            state["frame_idx"] = (state["frame_idx"] + 1) % total_frames
            _draw_frame(ax, run_summary, state["frame_idx"])
        return []

    ani = animation.FuncAnimation(fig, update, interval=interval_ms, blit=False, cache_frame_data=False)
    fig._motion_aware_animation = ani
    if state["paused"]:
        ani.event_source.stop()

    def on_key(event):
        if event.key in {" ", "space"}:
            state["paused"] = not state["paused"]
            if state["paused"]:
                ani.event_source.stop()
            else:
                ani.event_source.start()
            return
        if event.key in {"right", "n"}:
            state["paused"] = True
            ani.event_source.stop()
            state["frame_idx"] = min(total_frames - 1, state["frame_idx"] + 1)
        elif event.key in {"left", "p"}:
            state["paused"] = True
            ani.event_source.stop()
            state["frame_idx"] = max(0, state["frame_idx"] - 1)
        elif event.key == "home":
            state["paused"] = True
            ani.event_source.stop()
            state["frame_idx"] = 0
        elif event.key == "end":
            state["paused"] = True
            ani.event_source.stop()
            state["frame_idx"] = total_frames - 1
        else:
            return
        draw_current()

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
