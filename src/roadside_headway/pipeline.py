from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd

from roadside_headway.calibration.homography import Homography
from roadside_headway.calibration.lanes import LaneBoundaries
from roadside_headway.depth.depth_model import DepthEstimator
from roadside_headway.headway.compute import assign_lanes, compute_headway, compute_speed_per_track
from roadside_headway.headway.trajectory import extrapolate_constant_velocity
from roadside_headway.tracking.occlusion import occluded_track_ids
from roadside_headway.tracking.tracker import VehicleTracker


def run_pipeline(
    video_path: str | Path,
    homography: Homography,
    lanes: LaneBoundaries,
    direction: int = 1,
    use_depth: bool = True,
    start_frame: int = 0,
    max_frames: int | None = None,
    device: str = "mps",
    region_margin_frac: float = 1.0,
) -> pd.DataFrame:
    """Run detection + tracking + (optional) depth-assisted occlusion handling +
    homography projection over a video, then compute per-lane headway and
    per-track speed. Returns a long-format DataFrame, one row per (frame, track).

    `start_frame`/`max_frames` select a frame range — e.g. to evaluate on
    frames held out from an auto-calibration fit that used an earlier range.

    Detections whose pixel position lies far outside the region the
    homography was calibrated on are dropped entirely (see
    `Homography.is_within_calibrated_region`) — e.g. vehicles on an adjacent
    ramp visible in-frame but never in the calibration data, whose projected
    position can otherwise explode to nonsense (thousands of meters away)
    since a homography isn't reliable far past where it was fit.
    `region_margin_frac` controls how generous that bound is.
    """
    tracker = VehicleTracker(device=device)
    depth_estimator = DepthEstimator(device=device) if use_depth else None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    rows: list[dict] = []
    world_history: dict[int, list[tuple[float, float]]] = {}
    frame_idx = start_frame
    end_frame = None if max_frames is None else start_frame + max_frames

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if end_frame is not None and frame_idx >= end_frame:
                break

            detections = tracker.track_frame(frame)
            depth_map = depth_estimator.estimate(frame) if depth_estimator is not None else None
            occluded = occluded_track_ids(detections, depth_map) if depth_map is not None else set()

            for det in detections:
                if det.track_id is None:
                    continue
                tid = det.track_id
                history = world_history.setdefault(tid, [])

                if tid in occluded and len(history) >= 2:
                    world_pos = extrapolate_constant_velocity(history)
                else:
                    world_pos = homography.pixel_to_world(det.contact_point)

                if not homography.is_within_calibrated_region(world_pos, margin_frac=region_margin_frac):
                    continue
                history.append(world_pos)

                rows.append(
                    {
                        "frame": frame_idx,
                        "time_s": frame_idx / fps,
                        "track_id": tid,
                        "class_name": det.class_name,
                        "world_x": world_pos[0],
                        "world_y": world_pos[1],
                        "occluded": tid in occluded,
                    }
                )
            frame_idx += 1
    finally:
        cap.release()

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = assign_lanes(df, lanes)
    df = compute_headway(df, direction=direction)
    df = compute_speed_per_track(df)
    return df
