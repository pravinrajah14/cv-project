import pytest

from roadside_headway.evaluation.ngsim_sites import nominal_epoch_range_ms


def test_nominal_epoch_range_matches_known_us101_anchor():
    # Cross-checked by hand against the live ground-truth data: the earliest
    # Global_Time for us-101 is 1118846979700 ms, which is 2005-06-15 07:49:39.7
    # Pacific — a bit before the nominal 07:50:00 label, so it must fall
    # inside the padded range.
    start_ms, end_ms = nominal_epoch_range_ms("us-101", "0750am-0805am", buffer_s=90)
    known_earliest_global_time = 1118846979700
    assert start_ms <= known_earliest_global_time <= end_ms


def test_nominal_epoch_range_orders_start_before_end():
    start_ms, end_ms = nominal_epoch_range_ms("us-101", "0805am-0820am")
    assert start_ms < end_ms


def test_nominal_epoch_range_unknown_location_raises():
    with pytest.raises(KeyError):
        nominal_epoch_range_ms("nowhere", "0750am-0805am")
