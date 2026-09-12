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

        This is an approximation (varies with vehicle type/pose/camera angle) —
        see the README limitations section.
        """
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y2)


class VehicleDetector:
    """Single-frame vehicle detector, restricted to vehicle classes.

    Used directly for calibration/inspection tooling. The main pipeline uses
    `tracking.tracker.VehicleTracker` instead, which wraps the same model with
    persistent ID tracking across frames.
    """

    def __init__(self, model_name: str = "yolo11n.pt", device: str = "mps", conf: float = 0.25):
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
