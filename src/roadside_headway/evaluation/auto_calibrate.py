from __future__ import annotations

import numpy as np

from roadside_headway.calibration.homography import Homography


def _nearest_neighbor_pairs(
    pixel_points: list[tuple[float, float]],
    world_points: list[tuple[float, float]],
    homography: Homography,
    max_dist_m: float,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    if not pixel_points or not world_points:
        return [], []
    world_arr = np.array(world_points, dtype=float)
    matched_pixels: list[tuple[float, float]] = []
    matched_world: list[tuple[float, float]] = []
    for px in pixel_points:
        pred = np.array(homography.pixel_to_world(px), dtype=float)
        dists = np.linalg.norm(world_arr - pred, axis=1)
        idx = int(np.argmin(dists))
        if dists[idx] <= max_dist_m:
            matched_pixels.append(px)
            matched_world.append(world_points[idx])
    return matched_pixels, matched_world


def _corner_guess(
    px_min: np.ndarray, px_max: np.ndarray, w_min: np.ndarray, w_max: np.ndarray, flip_x: bool, flip_y: bool
) -> Homography:
    src = [
        (px_min[0], px_min[1]),
        (px_max[0], px_min[1]),
        (px_min[0], px_max[1]),
        (px_max[0], px_max[1]),
    ]
    wx0, wx1 = (w_max[0], w_min[0]) if flip_x else (w_min[0], w_max[0])
    wy0, wy1 = (w_max[1], w_min[1]) if flip_y else (w_min[1], w_max[1])
    dst = [(wx0, wy0), (wx1, wy0), (wx0, wy1), (wx1, wy1)]
    return Homography.fit(src, dst)


def fit_homography_by_matching(
    frames_pixel_points: dict[int, list[tuple[float, float]]],
    frames_world_points: dict[int, list[tuple[float, float]]],
    iterations: int = 6,
    initial_threshold_m: float = 25.0,
    final_threshold_m: float = 5.0,
) -> tuple[Homography, dict]:
    """Fit a homography from many (pixel, ground-truth) point sets without
    requiring a pre-established point-by-point correspondence.

    This is an iterative-closest-point (ICP) style procedure: given, for each
    of several frames, the set of pixel positions we detected and the set of
    ground-truth world positions actually present at that same instant (but
    with no known pairing between the two sets), it alternates between (a)
    matching each pixel point to its nearest ground-truth point under the
    *current* homography guess, within a shrinking distance threshold, and (b)
    refitting the homography on the matched pairs. It tries all 4 axis-flip
    combinations as starting guesses (pixel vs. world axes can point in either
    direction) and keeps whichever converges to the lowest residual.

    Useful whenever ground truth is available for the same footage (as with
    NGSIM) — it uses far more correspondences than a person could reasonably
    click by hand, and doesn't require guessing which lane-marking pixel
    corresponds to which real-world coordinate.
    """
    all_px = np.array([p for pts in frames_pixel_points.values() for p in pts], dtype=float)
    all_w = np.array([p for pts in frames_world_points.values() for p in pts], dtype=float)
    if len(all_px) < 4 or len(all_w) < 4:
        raise ValueError("Need at least 4 pixel points and 4 world points across all frames")

    px_min, px_max = all_px.min(axis=0), all_px.max(axis=0)
    w_min, w_max = all_w.min(axis=0), all_w.max(axis=0)
    thresholds = np.linspace(initial_threshold_m, final_threshold_m, iterations)

    best_homography: Homography | None = None
    best_stats: dict = {"mean_error_m": float("inf"), "n_matched": 0}

    for flip_x in (False, True):
        for flip_y in (False, True):
            homography = _corner_guess(px_min, px_max, w_min, w_max, flip_x, flip_y)
            stats = {"mean_error_m": float("inf"), "n_matched": 0}

            for threshold in thresholds:
                matched_px: list[tuple[float, float]] = []
                matched_w: list[tuple[float, float]] = []
                for frame_idx, pixel_points in frames_pixel_points.items():
                    world_points = frames_world_points.get(frame_idx, [])
                    px_pairs, w_pairs = _nearest_neighbor_pairs(pixel_points, world_points, homography, threshold)
                    matched_px.extend(px_pairs)
                    matched_w.extend(w_pairs)

                if len(matched_px) < 4:
                    break

                homography = Homography.fit(matched_px, matched_w)
                errors = [
                    float(np.hypot(*(np.array(homography.pixel_to_world(px)) - np.array(w))))
                    for px, w in zip(matched_px, matched_w)
                ]
                stats = {"mean_error_m": float(np.mean(errors)), "n_matched": len(matched_px)}

            if stats["mean_error_m"] < best_stats["mean_error_m"]:
                best_homography = homography
                best_stats = stats

    if best_homography is None:
        raise RuntimeError("Failed to converge on any axis-orientation guess")
    return best_homography, best_stats
