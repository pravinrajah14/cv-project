"""Generate a YOLO-format training set by auto-labeling video frames: run our
own detector/tracker, keep only the boxes whose projected position (via an
already-calibrated homography) lands near a real ground-truth vehicle at that
same instant, and use those confirmed detections' real bounding boxes as
labels. Frames get no manual annotation at all.

This exists because NGSIM's ground truth gives vehicle *positions*
(Local_X/Local_Y), not pixel bounding boxes, so there's no off-the-shelf
labeled set for this camera angle to fine-tune on. What this script produces
instead is weaker than hand-labeled boxes (it can only confirm vehicles the
detector already roughly finds — see README limitations) but costs nothing to
generate and is a real, high-quality dataset for this exact camera and
scene, not a generic one — good for tightening box regression and cutting
false positives, but not for finding vehicles the base model already misses.

Requires an existing homography for this camera view (e.g. from
scripts/calibrate_from_gt.py) — used here only to confirm/reject detections
against ground truth, not refit.

KNOWN BIAS (found empirically, see README Results): --confirm-threshold-m is
a *fixed* radius, but faster vehicles drift further between the exact
detection instant and ground truth's 0.1s-quantized timestamp than slow ones
do, so a fixed radius confirms slow/congested traffic more reliably than
fast/free-flowing traffic. Fine-tuning on the unmodified output of this
script measurably skewed a model toward slow-traffic conditions -- much
better accuracy there, much worse vehicle coverage overall. Before relying
on this for real fine-tuning, consider a velocity-scaled confirm threshold
or explicitly resampling for balanced speed coverage across the training set.

Usage:
    python scripts/generate_training_labels.py data/sb-camera4-0750am-0805am.avi \\
        --camera 4 --gt-csv data/us101_trajectories_0750am-0805am.csv \\
        --video-start-epoch-ms 1118846979700 --homography results/homography.json \\
        --num-frames 3000 --out datasets/ngsim_cam4
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from roadside_headway.calibration.homography import Homography
from roadside_headway.evaluation.ngsim_camera_coverage import collect_world_points_by_frame
from roadside_headway.tracking.tracker import VehicleTracker

CLASS_ID = 0  # single class: "vehicle"


def nearest_gt_distance(world_point: tuple[float, float], gt_points: list[tuple[float, float]]) -> float:
    if not gt_points:
        return float("inf")
    gt_arr = np.array(gt_points)
    return float(np.min(np.linalg.norm(gt_arr - np.array(world_point), axis=1)))


def bbox_to_yolo_line(bbox: tuple[float, float, float, float], img_w: int, img_h: int) -> str:
    x1, y1, x2, y2 = bbox
    x1, x2 = max(0.0, min(x1, img_w)), max(0.0, min(x2, img_w))
    y1, y2 = max(0.0, min(y1, img_h)), max(0.0, min(y2, img_h))
    x_center, y_center = (x1 + x2) / 2 / img_w, (y1 + y2) / 2 / img_h
    width, height = (x2 - x1) / img_w, (y2 - y1) / img_h
    return f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video")
    parser.add_argument("--camera", type=int, required=True)
    parser.add_argument("--gt-csv", required=True)
    parser.add_argument("--video-start-epoch-ms", type=int, required=True)
    parser.add_argument("--homography", default="results/homography.json")
    parser.add_argument("--num-frames", type=int, default=3000)
    parser.add_argument(
        "--confirm-threshold-m",
        type=float,
        default=4.0,
        help="Max distance (meters) between a projected detection and the nearest ground-truth "
        "vehicle for that detection to be confirmed as a real positive and kept as a label",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--out", default="datasets/ngsim_cam4")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    homography = Homography.load(args.homography)
    frames_world_points = collect_world_points_by_frame(
        args.gt_csv, args.camera, args.video_start_epoch_ms, args.num_frames
    )

    tracker = VehicleTracker(device=args.device)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")

    out_dir = Path(args.out)
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    n_confirmed = 0
    n_rejected = 0
    n_images = 0

    frame_idx = 0
    try:
        while frame_idx < args.num_frames:
            ok, frame = cap.read()
            if not ok:
                break
            detections = tracker.track_frame(frame)
            gt_points = frames_world_points.get(frame_idx, [])

            confirmed_lines = []
            for det in detections:
                world_pos = homography.pixel_to_world(det.contact_point)
                if nearest_gt_distance(world_pos, gt_points) <= args.confirm_threshold_m:
                    h, w = frame.shape[:2]
                    confirmed_lines.append(bbox_to_yolo_line(det.bbox, w, h))
                    n_confirmed += 1
                else:
                    n_rejected += 1

            split = "val" if rng.random() < args.val_fraction else "train"
            stem = f"frame{frame_idx:06d}"
            cv2.imwrite(str(out_dir / "images" / split / f"{stem}.jpg"), frame)
            (out_dir / "labels" / split / f"{stem}.txt").write_text("\n".join(confirmed_lines))
            n_images += 1

            frame_idx += 1
            if frame_idx % 500 == 0:
                print(f"  ...{frame_idx}/{args.num_frames} frames, {n_confirmed} confirmed so far")
    finally:
        cap.release()

    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {out_dir.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: vehicle\n"
    )

    print(
        f"\nWrote {n_images} images to {out_dir} "
        f"({n_confirmed} confirmed boxes, {n_rejected} unconfirmed detections left unlabeled)"
    )
    print(f"Dataset config: {data_yaml}")


if __name__ == "__main__":
    main()
