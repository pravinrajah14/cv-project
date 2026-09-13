"""Automated calibration for NGSIM footage: fit a homography by matching our
own detector/tracker's pixel positions against ground-truth vehicle positions
present at the same instant — no manual point-clicking, and the result is
directly comparable to NGSIM's ground truth (unlike scripts/calibrate.py's
manual mode, whose world frame won't line up with NGSIM's Local_X/Y unless the
person calibrating deliberately uses GT positions as reference points).

Requires a camera-coverage polygon for the chosen camera number — see
`roadside_headway.evaluation.ngsim_camera_coverage` (currently only US-101 is
populated).

Usage:
    python scripts/calibrate_from_gt.py data/sb-camera4-0750am-0805am.avi \\
        --camera 4 --gt-csv data/us101_trajectories_0750am-0805am.csv \\
        --video-start-epoch-ms 1118846979700 --num-frames 900 \\
        --out results/homography.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import pandas as pd

from roadside_headway.evaluation.auto_calibrate import fit_homography_by_matching, resolve_direction_ambiguity
from roadside_headway.evaluation.ngsim_camera_coverage import rows_in_camera_view
from roadside_headway.tracking.tracker import VehicleTracker

FEET_TO_METERS = 0.3048


def collect_pixel_points(
    video_path: str, num_frames: int, device: str
) -> tuple[dict[int, list[tuple[float, float]]], dict[int, list[tuple[float, float]]]]:
    """Returns (frames_pixel_points, track_pixel_sequences): the former keyed
    by frame index (for ICP position matching), the latter by track id (for
    `resolve_direction_ambiguity`, which needs motion, not just position)."""
    tracker = VehicleTracker(device=device)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    frames_pixel_points: dict[int, list[tuple[float, float]]] = {}
    track_pixel_sequences: dict[int, list[tuple[float, float]]] = {}
    frame_idx = 0
    try:
        while frame_idx < num_frames:
            ok, frame = cap.read()
            if not ok:
                break
            detections = tracker.track_frame(frame)
            frames_pixel_points[frame_idx] = [d.contact_point for d in detections]
            for d in detections:
                if d.track_id is not None:
                    track_pixel_sequences.setdefault(d.track_id, []).append(d.contact_point)
            frame_idx += 1
    finally:
        cap.release()
    return frames_pixel_points, track_pixel_sequences


def collect_world_points(
    gt_csv: str, camera: int, video_start_epoch_ms: int, num_frames: int
) -> dict[int, list[tuple[float, float]]]:
    df = pd.read_csv(gt_csv)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df[rows_in_camera_view(df, camera)]

    frames_world_points: dict[int, list[tuple[float, float]]] = {}
    for frame_idx in range(num_frames):
        global_time = video_start_epoch_ms + frame_idx * 100
        rows = df[df["global_time"] == global_time]
        points = list(zip(rows["local_y"] * FEET_TO_METERS, rows["local_x"] * FEET_TO_METERS))
        if points:
            frames_world_points[frame_idx] = points
    return frames_world_points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video")
    parser.add_argument("--camera", type=int, required=True)
    parser.add_argument("--gt-csv", required=True)
    parser.add_argument("--video-start-epoch-ms", type=int, required=True)
    parser.add_argument("--num-frames", type=int, default=900, help="How many video frames to use (default: 90s)")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--out", default="results/homography.json")
    args = parser.parse_args()

    print(f"Tracking {args.num_frames} frames of {args.video} ...")
    frames_pixel_points, track_pixel_sequences = collect_pixel_points(args.video, args.num_frames, args.device)
    n_pixel_points = sum(len(v) for v in frames_pixel_points.values())
    print(f"  collected {n_pixel_points} pixel detections across {len(frames_pixel_points)} frames")

    print(f"Loading ground truth for camera {args.camera} ...")
    frames_world_points = collect_world_points(args.gt_csv, args.camera, args.video_start_epoch_ms, args.num_frames)
    n_world_points = sum(len(v) for v in frames_world_points.values())
    print(f"  found {n_world_points} ground-truth vehicle-instants in this camera's coverage")

    print("Fitting homography by iterative nearest-neighbor matching...")
    homography, stats = fit_homography_by_matching(frames_pixel_points, frames_world_points)
    print(f"  matched {stats['n_matched']} point pairs, mean residual {stats['mean_error_m']:.3f} m")

    print("Checking longitudinal direction against tracked vehicle motion...")
    before = homography.pixel_to_world((0, 0))
    homography = resolve_direction_ambiguity(homography, track_pixel_sequences)
    flipped = homography.pixel_to_world((0, 0)) != before
    print(f"  {'flipped' if flipped else 'no flip needed'} (position-only fitting can't tell direction of travel"
          " from a single snapshot, so this checks it separately using tracked motion)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    homography.save(out_path)

    calibration_meta = out_path.with_name(out_path.stem + "_calibration_meta.json")
    calibration_meta.write_text(
        json.dumps(
            {
                "method": "auto_calibrate.fit_homography_by_matching",
                "camera": args.camera,
                "num_frames": args.num_frames,
                "n_matched_points": stats["n_matched"],
                "mean_residual_m": stats["mean_error_m"],
                "direction_flip_applied": flipped,
                "world_frame": "NGSIM Local_Y (longitudinal, meters) = world_x; Local_X (lateral, meters) = world_y",
            },
            indent=2,
        )
    )
    print(f"\nSaved homography to {out_path} and calibration provenance to {calibration_meta}")


if __name__ == "__main__":
    main()
