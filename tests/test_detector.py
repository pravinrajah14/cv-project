from roadside_headway.detection.detector import Detection


def test_contact_point_is_bottom_center():
    det = Detection(bbox=(10, 20, 30, 40), confidence=0.9, class_name="car")
    assert det.contact_point == (20.0, 40.0)


def test_leading_edge_point_positive_direction_uses_right_edge():
    det = Detection(bbox=(10, 20, 30, 40), confidence=0.9, class_name="car")
    assert det.leading_edge_point(direction_sign=1) == (30.0, 30.0)


def test_leading_edge_point_negative_direction_uses_left_edge():
    det = Detection(bbox=(10, 20, 30, 40), confidence=0.9, class_name="car")
    assert det.leading_edge_point(direction_sign=-1) == (10.0, 30.0)
