from __future__ import annotations

import numpy as np

from roadside_headway.detection.detector import VEHICLE_CLASS_NAMES, Detection


class VehicleTracker:
    """Frame-by-frame multi-object tracking via Ultralytics' built-in ByteTrack.

    Call `track_frame` once per frame in order (it maintains internal state);
    call `reset()` before starting a new video.

    Defaults match `detection.detector.VehicleDetector` — see its docstring
    for why yolo11n (nano) doesn't work on NGSIM's steep overhead camera angle
    and yolo11s (small) at a low confidence threshold does.
    """

    def __init__(
        self,
        model_name: str = "yolo11s.pt",
        device: str = "mps",
        conf: float = 0.1,
        tracker_cfg: str = "bytetrack.yaml",
        vehicle_class_names: set[str] | None = None,
    ):
        from ultralytics import YOLO  # imported lazily so this module stays testable without ultralytics installed

        self.model = YOLO(model_name)
        self.device = device
        self.conf = conf
        self.tracker_cfg = tracker_cfg
        # Override for a custom-trained model whose class vocabulary isn't COCO's
        # (e.g. a single "vehicle" class from scripts/generate_training_labels.py) --
        # the default set matches nothing there, silently filtering out everything.
        names = vehicle_class_names if vehicle_class_names is not None else VEHICLE_CLASS_NAMES
        self._vehicle_class_ids = sorted(idx for idx, name in self.model.names.items() if name in names)
        if not self._vehicle_class_ids:
            raise ValueError(
                f"None of the model's classes ({list(self.model.names.values())}) matched "
                f"vehicle_class_names={names} -- pass the model's own class names explicitly."
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
