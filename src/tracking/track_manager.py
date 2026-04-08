"""Track manager for dynamic object hypotheses."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.perception.observation import Observation
from src.tracking.kalman_tracker import ConstantVelocityKalman, KalmanTrack


@dataclass
class TrackManager:
    tracker: ConstantVelocityKalman
    stale_frames: int
    tracks: dict[str, KalmanTrack] = field(default_factory=dict)
    next_track_id: int = 0
    id_switches: int = 0

    def predict_all(self) -> None:
        for track in self.tracks.values():
            self.tracker.predict(track)

    def active_candidates(self) -> list[tuple[str, str, np.ndarray]]:
        return [(track_id, track.label, track.state[:2].copy()) for track_id, track in self.tracks.items()]

    def update_track(self, track_id: str, observation: Observation) -> None:
        track = self.tracks[track_id]
        if track.object_id is not None and observation.object_id != track.object_id:
            self.id_switches += 1
        track.object_id = observation.object_id
        self.tracker.update(track, observation.world_position)

    def spawn_track(self, observation: Observation) -> str:
        track_id = f"trk_{self.next_track_id:03d}"
        self.next_track_id += 1
        self.tracks[track_id] = self.tracker.initialize(track_id=track_id, label=observation.label, measurement=observation.world_position, object_id=observation.object_id)
        return track_id

    def prune(self) -> None:
        stale_ids = [track_id for track_id, track in self.tracks.items() if track.missed_frames > self.stale_frames]
        for track_id in stale_ids:
            del self.tracks[track_id]
