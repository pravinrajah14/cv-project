import math

from roadside_headway.calibration.lanes import LaneBoundaries


def test_lane_of_assigns_correct_lane():
    lanes = LaneBoundaries(boundaries=[0.0, 3.5, 7.0, 10.5])
    assert lanes.lane_of(1.0) == 0
    assert lanes.lane_of(5.0) == 1
    assert lanes.lane_of(9.0) == 2


def test_lane_of_out_of_range_is_nan():
    lanes = LaneBoundaries(boundaries=[0.0, 3.5])
    assert math.isnan(lanes.lane_of(-1.0))
    assert math.isnan(lanes.lane_of(10.0))


def test_lane_of_handles_descending_boundaries():
    lanes = LaneBoundaries(boundaries=[10.5, 7.0, 3.5, 0.0])
    assert lanes.lane_of(9.0) == 0
    assert lanes.lane_of(1.0) == 2


def test_save_and_load_round_trip(tmp_path):
    lanes = LaneBoundaries(boundaries=[0.0, 3.5, 7.0])
    path = tmp_path / "lanes.json"
    lanes.save(path)
    loaded = LaneBoundaries.load(path)
    assert loaded.boundaries == lanes.boundaries
