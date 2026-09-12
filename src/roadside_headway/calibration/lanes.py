from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LaneBoundaries:
    """Lateral (world-frame) boundaries between lanes, in meters, ordered
    left to right (or right to left — only the ordering, not the direction,
    matters). `boundaries = [b0, b1, ..., bn]` defines n lanes: lane i spans
    the interval between boundaries[i] and boundaries[i+1].
    """

    boundaries: list[float]

    def lane_of(self, lateral: float) -> int | float:
        """Return the lane index for a lateral coordinate, or NaN if outside all lanes."""
        for i in range(len(self.boundaries) - 1):
            lo, hi = sorted((self.boundaries[i], self.boundaries[i + 1]))
            if lo <= lateral <= hi:
                return i
        return math.nan

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"boundaries": self.boundaries}, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "LaneBoundaries":
        data = json.loads(Path(path).read_text())
        return cls(boundaries=data["boundaries"])
