import math

import pandas as pd
import pytest

from roadside_headway.calibration.lanes import LaneBoundaries
from roadside_headway.headway.compute import assign_lanes, compute_headway, compute_speed_per_track


def test_assign_lanes():
    df = pd.DataFrame({"world_y": [1.0, 5.0, 9.0]})
    lanes = LaneBoundaries(boundaries=[0.0, 3.5, 7.0, 10.5])
    out = assign_lanes(df, lanes)
    assert list(out["lane"]) == [0, 1, 2]


def test_compute_headway_orders_by_progress_and_gives_correct_gaps():
    df = pd.DataFrame(
        {
            "frame": [0, 0, 0],
            "track_id": [1, 2, 3],
            "world_x": [0.0, 10.0, 20.0],
            "lane": [0, 0, 0],
        }
    )
    out = compute_headway(df, direction=1)

    row1 = out[out["track_id"] == 1].iloc[0]
    row2 = out[out["track_id"] == 2].iloc[0]
    row3 = out[out["track_id"] == 3].iloc[0]

    assert row1["headway_m"] == pytest.approx(10.0)
    assert row1["lead_track_id"] == 2
    assert row2["headway_m"] == pytest.approx(10.0)
    assert row2["lead_track_id"] == 3
    assert math.isnan(row3["headway_m"])  # nothing ahead of the lead vehicle


def test_compute_headway_respects_direction_sign():
    df = pd.DataFrame(
        {
            "frame": [0, 0],
            "track_id": [1, 2],
            "world_x": [0.0, 10.0],
            "lane": [0, 0],
        }
    )
    # direction=-1 means smaller world_x is "ahead"; vehicle 1 (x=0) is the leader
    out = compute_headway(df, direction=-1)
    row2 = out[out["track_id"] == 2].iloc[0]
    assert row2["headway_m"] == pytest.approx(10.0)
    assert row2["lead_track_id"] == 1


def test_compute_headway_ignores_rows_outside_any_lane():
    df = pd.DataFrame(
        {
            "frame": [0, 0],
            "track_id": [1, 2],
            "world_x": [0.0, 10.0],
            "lane": [float("nan"), float("nan")],
        }
    )
    out = compute_headway(df, direction=1)
    assert out["headway_m"].isna().all()


def test_compute_speed_per_track_constant_velocity():
    df = pd.DataFrame(
        {
            "track_id": [1] * 10,
            "time_s": list(range(10)),
            "world_x": [3.0 * t for t in range(10)],
            "world_y": [0.0] * 10,
        }
    )
    out = compute_speed_per_track(df, smoothing_window=3)
    valid = out["speed_mps"].dropna()
    assert (valid > 2.9).all() and (valid < 3.1).all()
