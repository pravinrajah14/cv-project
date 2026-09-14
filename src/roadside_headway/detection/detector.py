from __future__ import annotations

from dataclasses import dataclass

import numpy as np

VEHICLE_CLASS_NAMES = {"car", "truck", "bus", "motorcycle"}


@dataclass
class Detection:
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    confidence: float
    class_name: str
    track_id: int | None = None

    @property
    def contact_point(self) -> tuple[float, float]:
        """Bottom-center of the box: proxy for the vehicle's ground-contact point.

        Designed for a typical oblique roadside camera, where the bottom edge
        of the box is the part of the vehicle closest to the camera. For a
        near-overhead camera (like NGSIM's) this is the wrong geometric
        intuition — see `leading_edge_point` below, which is a better fit for
        that case specifically.

        This is an approximation (varies with vehicle type/pose/camera angle) —
        see the README limitations section.
        """
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y2)

    def leading_edge_point(self, direction_sign: int) -> tuple[float, float]:
        """The box's leading edge (in the direction of travel) at vertical
        center — for a near-overhead camera where pixel-x is roughly
        longitudinal and pixel-y roughly lateral, this is a much closer
        geometric match to NGSIM's front-center ground-truth definition than
        `contact_point`: it's centered on the lane laterally (not offset by
        half the box height, as bottom-center is) and picks the front rather
        than an arbitrary edge. `direction_sign` is +1 if vehicles move
        toward increasing pixel-x on screen, -1 otherwise — see
        `auto_calibrate.estimate_pixel_direction_sign`.
        """
        x1, y1, x2, y2 = self.bbox
        y_center = (y1 + y2) / 2.0
        return (x2, y_center) if direction_sign > 0 else (x1, y_center)


class VehicleDetector:
    """Single-frame vehicle detector, restricted to vehicle classes.

    Used directly for calibration/inspection tooling. The main pipeline uses
    `tracking.tracker.VehicleTracker` instead, which wraps the same model with
    persistent ID tracking across frames.

    Defaults were tuned empirically against the NGSIM overhead camera view:
    yolo11n (nano) essentially fails to detect any vehicles at all from this
    steep top-down angle (cars are ~15-30px, and COCO's training distribution
    is overwhelmingly street-level/oblique views) — yolo11s (small) at a low
    confidence threshold does detect them, still comfortably lightweight on an
    M3. Different footage (resolution, camera angle, vehicle size) may need
    re-tuning; see README limitations.
    """

    def __init__(self, model_name: str = "yolo11s.pt", device: str = "mps", conf: float = 0.1):
        from ultralytics import YOLO  # imported lazily so this module stays testable without ultralytics installed

        self.model = YOLO(model_name)
        self.device = device
        self.conf = conf
        self._vehicle_class_ids = sorted(
            idx for idx, name in self.model.names.items() if name in VEHICLE_CLASS_NAMES
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            frame, device=self.device, conf=self.conf, classes=self._vehicle_class_ids, verbose=False
        )
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=float(box.conf.item()),
                        class_name=self.model.names[cls_id],
                    )
                )
        return detections
