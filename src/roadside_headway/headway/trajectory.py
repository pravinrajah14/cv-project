from __future__ import annotations

import numpy as np


def extrapolate_constant_velocity(history: list[tuple[float, float]]) -> tuple[float, float]:
    """Predict the next world-space position assuming constant velocity from
    the last two known positions. Used to fill in a position for a frame where
    the vehicle's true contact point is occluded (see tracking.occlusion)."""
    (x0, y0), (x1, y1) = history[-2], history[-1]
    return (2 * x1 - x0, 2 * y1 - y0)


def smooth_positions(positions: np.ndarray, window: int = 5) -> np.ndarray:
    """Centered moving-average smoothing along axis 0, to reduce pixel-quantization
    jitter before differentiating for speed. positions: (N, 2). Leaves the first/last
    `window // 2` samples on each end unsmoothed rather than distorting them."""
    if window < 2 or len(positions) < window:
        return positions.copy()
    kernel = np.ones(window) / window
    smoothed = np.empty_like(positions)
    for col in range(positions.shape[1]):
        smoothed[:, col] = np.convolve(positions[:, col], kernel, mode="same")
    half = window // 2
    if half > 0:
        smoothed[:half] = positions[:half]
        smoothed[-half:] = positions[-half:]
    return smoothed


def compute_speed(times: np.ndarray, longitudinal: np.ndarray) -> np.ndarray:
    """Central-difference speed (m/s) from (smoothed) longitudinal position over time.

    Only the longitudinal component is used — a simplification appropriate for
    through-traffic in a lane, not for lane-change maneuvers. Flagged in the
    README limitations."""
    if len(times) < 2:
        return np.full_like(longitudinal, np.nan, dtype=float)
    return np.gradient(longitudinal, times)
