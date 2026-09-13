"""Derive lane lateral boundaries directly from ground-truth vehicle
positions, for use with a homography calibrated by scripts/calibrate_from_gt.py
(whose world_y already matches NGSIM's Local_X convention, in meters).

For each Lane_ID present in a camera's coverage, takes the median lateral
position of vehicles assigned to it, then sets boundaries at the midpoints
between consecutive lane medians (and half a lane-pitch beyond the outermost
medians) — more reliable than reading faint lane-marking pixels off a
compressed, low-resolution video frame.

Usage:
    python scripts/derive_lanes_from_gt.py --camera 4 \\
        --gt-csv data/us101_trajectories_0750am-0805am.csv \\
        --out results/lanes.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from roadside_headway.calibration.lanes import LaneBoundaries
from roadside_headway.evaluation.ngsim_camera_coverage import rows_in_camera_view

FEET_TO_METERS = 0.3048


def derive_boundaries(gt_csv: str, camera: int) -> list[float]:
    df = pd.read_csv(gt_csv)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df[rows_in_camera_view(df, camera)].copy()
    df["local_x_m"] = df["local_x"] * FEET_TO_METERS

    medians = df.groupby("lane_id")["local_x_m"].median().sort_index()
    if len(medians) < 2:
        raise ValueError(f"Only found {len(medians)} lane(s) in camera {camera}'s coverage — need at least 2")

    values = medians.to_numpy()
    half_pitch_low = (values[1] - values[0]) / 2
    half_pitch_high = (values[-1] - values[-2]) / 2

    boundaries = [values[0] - half_pitch_low]
    boundaries.extend((values[i] + values[i + 1]) / 2 for i in range(len(values) - 1))
    boundaries.append(values[-1] + half_pitch_high)
    return boundaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, required=True)
    parser.add_argument("--gt-csv", required=True)
    parser.add_argument("--out", default="results/lanes.json")
    args = parser.parse_args()

    boundaries = derive_boundaries(args.gt_csv, args.camera)
    print(f"Derived {len(boundaries) - 1} lane(s), boundaries (m): {[round(b, 3) for b in boundaries]}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    LaneBoundaries(boundaries=boundaries).save(out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
