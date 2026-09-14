import pytest

from roadside_headway.detection.detector import VehicleDetector
from roadside_headway.tracking.tracker import VehicleTracker


def test_tracker_default_matches_coco_vehicle_classes():
    tracker = VehicleTracker(model_name="yolo11s.pt", device="cpu")
    assert tracker._vehicle_class_ids  # car/truck/bus/motorcycle should all resolve in COCO


def test_tracker_raises_when_no_class_names_match():
    with pytest.raises(ValueError, match="vehicle_class_names"):
        VehicleTracker(model_name="yolo11s.pt", device="cpu", vehicle_class_names={"vehicle"})


def test_tracker_accepts_custom_class_names_present_in_model():
    # "person" exists in stock COCO weights -- stands in for a fine-tuned
    # model's own custom class name without needing a real fine-tuned model.
    tracker = VehicleTracker(model_name="yolo11s.pt", device="cpu", vehicle_class_names={"person"})
    assert tracker._vehicle_class_ids


def test_detector_raises_when_no_class_names_match():
    with pytest.raises(ValueError, match="vehicle_class_names"):
        VehicleDetector(model_name="yolo11s.pt", device="cpu", vehicle_class_names={"vehicle"})
