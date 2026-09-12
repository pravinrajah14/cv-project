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
    def fit(cls, pixel_points: list[tuple[float, float]], world_points: list[tuple[float, float]]) -> "Homography":
        if len(pixel_points) < 4 or len(pixel_points) != len(world_points):
            raise ValueError("Need >= 4 matched pixel/world point pairs of equal length")
        src = np.array(pixel_points, dtype=np.float64)
        dst = np.array(world_points, dtype=np.float64)
        matrix, _ = cv2.findHomography(src, dst, method=0)
        if matrix is None:
            raise ValueError("cv2.findHomography failed to converge on the given points")
        return cls(matrix=matrix, source_points=list(pixel_points), dest_points=list(world_points))

    def pixel_to_world(self, point: tuple[float, float]) -> tuple[float, float]:
        px = np.array([point[0], point[1], 1.0])
        world = self.matrix @ px
        world = world / world[2]
        return float(world[0]), float(world[1])

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
