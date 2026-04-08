"""Constant-velocity Kalman tracker."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KalmanTrack:
    track_id: str
    label: str
    state: np.ndarray
    covariance: np.ndarray
    age: int = 1
    missed_frames: int = 0
    object_id: str | None = None


class ConstantVelocityKalman:
    def __init__(self, process_var: float, measurement_var: float) -> None:
        self.process_var = process_var
        self.measurement_var = measurement_var
        self.h = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)

    def _f(self, dt: float) -> np.ndarray:
        return np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)

    def _q(self, dt: float) -> np.ndarray:
        q = self.process_var
        return q * np.array([
            [dt**4 / 4, 0, dt**3 / 2, 0],
            [0, dt**4 / 4, 0, dt**3 / 2],
            [dt**3 / 2, 0, dt**2, 0],
            [0, dt**3 / 2, 0, dt**2],
        ], dtype=float)

    def initialize(self, track_id: str, label: str, measurement: np.ndarray, object_id: str | None = None) -> KalmanTrack:
        state = np.array([measurement[0], measurement[1], 0.0, 0.0], dtype=float)
        covariance = np.eye(4, dtype=float)
        return KalmanTrack(track_id=track_id, label=label, state=state, covariance=covariance, object_id=object_id)

    def predict(self, track: KalmanTrack, dt: float = 1.0) -> None:
        f = self._f(dt)
        track.state = f @ track.state
        track.covariance = f @ track.covariance @ f.T + self._q(dt)
        track.age += 1
        track.missed_frames += 1

    def update(self, track: KalmanTrack, measurement: np.ndarray) -> None:
        r = np.eye(2, dtype=float) * self.measurement_var
        innovation = measurement - self.h @ track.state
        s = self.h @ track.covariance @ self.h.T + r
        k = track.covariance @ self.h.T @ np.linalg.inv(s)
        track.state = track.state + k @ innovation
        track.covariance = (np.eye(4, dtype=float) - k @ self.h) @ track.covariance
        track.missed_frames = 0

    def association_cost(self, track: KalmanTrack, measurement: np.ndarray) -> float:
        return float(np.linalg.norm(track.state[:2] - measurement))
