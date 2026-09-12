import numpy as np

from roadside_headway.detection.detector import Detection
from roadside_headway.tracking.occlusion import iou, occluded_track_ids


def test_iou_no_overlap_is_zero():
    assert iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0.0


def test_iou_identical_boxes_is_one():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_partial_overlap():
    # two 10x10 boxes overlapping in a 5x10 region -> intersection 50, union 150
    value = iou((0, 0, 10, 10), (5, 0, 15, 10))
    assert abs(value - 50 / 150) < 1e-9


def test_occluded_track_ids_flags_the_farther_vehicle():
    a = Detection(bbox=(0, 0, 10, 10), confidence=0.9, class_name="car", track_id=1)
    b = Detection(bbox=(5, 0, 15, 10), confidence=0.9, class_name="car", track_id=2)
    depth_map = np.zeros((10, 15), dtype=np.float32)
    depth_map[:, :10] = 5.0  # track 1's region: farther (smaller value = farther, per Depth Anything convention)
    depth_map[:, 10:] = 20.0  # track 2's region: nearer

    occluded = occluded_track_ids([a, b], depth_map, iou_threshold=0.05)
    assert occluded == {1}


def test_occluded_track_ids_ignores_non_overlapping_boxes():
    a = Detection(bbox=(0, 0, 5, 5), confidence=0.9, class_name="car", track_id=1)
    b = Detection(bbox=(50, 50, 55, 55), confidence=0.9, class_name="car", track_id=2)
    depth_map = np.ones((60, 60), dtype=np.float32)

    occluded = occluded_track_ids([a, b], depth_map)
    assert occluded == set()
