from __future__ import annotations

import numpy as np

from roadside_headway.detection.detector import VEHICLE_CLASS_NAMES, Detection


class VehicleTracker:
    """Frame-by-frame multi-object tracking via Ultralytics' built-in ByteTrack.

    Call `track_frame` once per frame in order (it maintains internal state);
    call `reset()` before starting a new video.
    """

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        device: str = "mps",
        conf: float = 0.25,
        tracker_cfg: str = "bytetrack.yaml",
    ):
        from ultralytics import YOLO  # imported lazily so this module stays testable without ultralytics installed

        self.model = YOLO(model_name)
        self.device = device
        self.conf = conf
        self.tracker_cfg = tracker_cfg
        self._vehicle_class_ids = sorted(
            idx for idx, name in self.model.names.items() if name in VEHICLE_CLASS_NAMES
        )

    def track_frame(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.track(
            frame,
            device=self.device,
            conf=self.conf,
            classes=self._vehicle_class_ids,
            tracker=self.tracker_cfg,
            persist=True,
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            if result.boxes is None or result.boxes.id is None:
                continue
            for box, track_id in zip(result.boxes, result.boxes.id):
                cls_id = int(box.cls.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=float(box.conf.item()),
                        class_name=self.model.names[cls_id],
                        track_id=int(track_id.item()),
                    )
                )
        return detections

    def reset(self) -> None:
        """Clear tracker state before starting a new video."""
        predictor = getattr(self.model, "predictor", None)
        trackers = getattr(predictor, "trackers", None) if predictor is not None else None
        if trackers:
            for t in trackers:
                t.reset()
