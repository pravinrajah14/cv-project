import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from generate_training_labels import bbox_to_yolo_line, nearest_gt_distance  # noqa: E402


def test_nearest_gt_distance_finds_closest_point():
    gt_points = [(0.0, 0.0), (10.0, 0.0), (3.0, 4.0)]
    assert nearest_gt_distance((3.0, 4.0), gt_points) == 0.0
    assert nearest_gt_distance((0.0, 1.0), gt_points) == 1.0


def test_nearest_gt_distance_empty_list_is_infinite():
    assert nearest_gt_distance((0.0, 0.0), []) == float("inf")


def test_bbox_to_yolo_line_centers_and_normalizes():
    # bbox spans the whole 100x50 image
    line = bbox_to_yolo_line((0.0, 0.0, 100.0, 50.0), img_w=100, img_h=50)
    class_id, x, y, w, h = line.split()
    assert class_id == "0"
    assert float(x) == 0.5
    assert float(y) == 0.5
    assert float(w) == 1.0
    assert float(h) == 1.0


def test_bbox_to_yolo_line_clips_to_image_bounds():
    # bbox partially outside the image on all sides
    line = bbox_to_yolo_line((-10.0, -10.0, 110.0, 60.0), img_w=100, img_h=50)
    _, x, y, w, h = line.split()
    assert float(w) == 1.0  # clipped back to full width
    assert float(h) == 1.0  # clipped back to full height
