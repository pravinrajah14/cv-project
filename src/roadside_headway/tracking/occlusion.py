from __future__ import annotations

import numpy as np

from roadside_headway.depth.depth_model import median_depth_in_box
from roadside_headway.detection.detector import Detection


def iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def occluded_track_ids(
    detections: list[Detection], depth_map: np.ndarray, iou_threshold: float = 0.05
) -> set[int]:
    """Flag the rear (occluded) vehicle in each pair of overlapping boxes.

    Depth Anything's convention is larger value = nearer the camera, so for an
    overlapping pair, the box with the smaller median depth is farther away
    and treated as (partially) occluded by the nearer one — regardless of
    which one is "ahead" along the road, since optical occlusion depends only
    on camera-relative depth. Callers should fall back to track-history
    extrapolation for flagged tracks instead of trusting their visible
    (possibly clipped) contact point this frame.
    """
    occluded: set[int] = set()
    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            a, b = detections[i], detections[j]
            if a.track_id is None or b.track_id is None:
                continue
            if iou(a.bbox, b.bbox) < iou_threshold:
                continue
            depth_a = median_depth_in_box(depth_map, a.bbox)
            depth_b = median_depth_in_box(depth_map, b.bbox)
            if np.isnan(depth_a) or np.isnan(depth_b):
                continue
            rear = a if depth_a < depth_b else b
            occluded.add(rear.track_id)
    return occluded
