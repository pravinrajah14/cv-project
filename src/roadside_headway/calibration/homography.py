from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Homography:
    """Maps image pixel coordinates to real-world ground-plane coordinates (meters).

    World coordinates are (s, l): s = longitudinal position along the road
    direction, l = lateral offset across lanes. Fit from >= 4 point
    correspondences between pixel coordinates (clicked on a reference frame)
    and real-world (s, l) measured on site — e.g. from standard lane-marking
    dimensions, or from a published site diagram/metadata for the footage.

    Assumes the road is planar. Any grade or point-picking error biases
    longitudinal estimates, worse further from the camera (foreshortening) —
    see the README limitations section.
    """

    matrix: np.ndarray  # 3x3, pixel (homogeneous) -> world (homogeneous)
    source_points: list[tuple[float, float]] = field(default_factory=list)
    dest_points: list[tuple[float, float]] = field(default_factory=list)

    @classmethod
    def fit(
        cls,
        pixel_points: list[tuple[float, float]],
        world_points: list[tuple[float, float]],
        robust: bool = False,
        ransac_threshold_m: float = 5.0,
    ) -> "Homography":
        """Fit a homography via plain least squares by default (deterministic
        — the right choice for an exact manual calibration, and for stable
        convergence inside `auto_calibrate`'s iterative matching loop).

        Pass `robust=True` to fit with RANSAC instead, which rejects outlier
        correspondences: important when the pairing itself was inferred (as
        in `auto_calibrate.fit_homography_by_matching`'s final fit) and can
        include some wrong matches even after distance thresholding. A
        handful of bad pairs can otherwise skew plain least squares into a
        nearly-singular matrix that fits the training points fine but
        explodes for pixel positions outside their immediate neighborhood
        (observed directly on real data: det() ~ -3.6e-4 from a plain fit
        that still had ~3m mean residual on its own training points).
        `ransac_threshold_m` is the max reprojection error, in world units,
        for a pair to count as an inlier; with `robust=True`, outlier pairs
        are dropped from `source_points`/`dest_points` so
        `reprojection_error_m` and `calibrated_bounds` reflect only the
        points actually used.
        """
        if len(pixel_points) < 4 or len(pixel_points) != len(world_points):
            raise ValueError("Need >= 4 matched pixel/world point pairs of equal length")
        src = np.array(pixel_points, dtype=np.float64)
        dst = np.array(world_points, dtype=np.float64)
        method = cv2.RANSAC if (robust and len(pixel_points) > 4) else 0
        matrix, mask = cv2.findHomography(src, dst, method=method, ransacReprojThreshold=ransac_threshold_m)
        if matrix is None:
            raise ValueError("cv2.findHomography failed to converge on the given points")
        if mask is not None and method == cv2.RANSAC:
            inliers = mask.ravel().astype(bool)
            pixel_points = [p for p, keep in zip(pixel_points, inliers) if keep]
            world_points = [p for p, keep in zip(world_points, inliers) if keep]
        return cls(matrix=matrix, source_points=list(pixel_points), dest_points=list(world_points))

    def pixel_to_world(self, point: tuple[float, float]) -> tuple[float, float]:
        px = np.array([point[0], point[1], 1.0])
        world = self.matrix @ px
        world = world / world[2]
        return float(world[0]), float(world[1])

    def calibrated_bounds(self, margin_frac: float = 1.0) -> tuple[tuple[float, float], tuple[float, float]]:
        """Bounding box of the world points this homography was fit on,
        expanded by `margin_frac` of each axis's span on both sides. A
        homography is only reliable near the region it was calibrated on —
        projecting a pixel far outside that region (e.g. a vehicle on an
        adjacent ramp visible in-frame but outside the calibrated lanes) can
        blow up arbitrarily, especially near the transform's vanishing line.
        Used by `is_within_calibrated_region` as a sanity bound, not a precise
        field-of-view mask.
        """
        if not self.dest_points:
            return (-np.inf, np.inf), (-np.inf, np.inf)
        xs = [p[0] for p in self.dest_points]
        ys = [p[1] for p in self.dest_points]
        x_span = max(xs) - min(xs)
        y_span = max(ys) - min(ys)
        return (
            (min(xs) - margin_frac * x_span, max(xs) + margin_frac * x_span),
            (min(ys) - margin_frac * y_span, max(ys) + margin_frac * y_span),
        )

    def is_within_calibrated_region(self, point: tuple[float, float], margin_frac: float = 1.0) -> bool:
        (x_lo, x_hi), (y_lo, y_hi) = self.calibrated_bounds(margin_frac)
        x, y = point
        return x_lo <= x <= x_hi and y_lo <= y <= y_hi

    def reprojection_error_m(self) -> float:
        """Mean Euclidean error (meters) of the fitted homography on its own
        calibration points — a lower bound on calibration accuracy, not a full
        validation (that requires held-out points or the BrnoCompSpeed check)."""
        if not self.source_points:
            return float("nan")
        errors = []
        for src, dst in zip(self.source_points, self.dest_points):
            pred_x, pred_y = self.pixel_to_world(src)
            dst_x, dst_y = dst
            errors.append(float(np.hypot(pred_x - dst_x, pred_y - dst_y)))
        return float(np.mean(errors))

    def save(self, path: str | Path) -> None:
        data = {
            "matrix": self.matrix.tolist(),
            "source_points": self.source_points,
            "dest_points": self.dest_points,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Homography":
        data = json.loads(Path(path).read_text())
        return cls(
            matrix=np.array(data["matrix"], dtype=np.float64),
            source_points=[tuple(p) for p in data["source_points"]],
            dest_points=[tuple(p) for p in data["dest_points"]],
        )
