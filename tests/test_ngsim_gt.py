import pandas as pd
import pytest

from roadside_headway.evaluation.ngsim_gt import compute_metrics, match_tracks_to_ground_truth


def test_match_tracks_to_ground_truth_picks_nearest_and_attaches_columns():
    pred_df = pd.DataFrame(
        {
            "frame": [0, 0],
            "time_s": [0.0, 0.0],
            "track_id": [1, 2],
            "world_x": [0.0, 10.0],
            "world_y": [0.0, 0.0],
        }
    )
    gt_df = pd.DataFrame(
        {
            "vehicle_id": [101, 102, 999],
            "time_s": [0.0, 0.0, 0.0],
            "gt_x_m": [0.1, 10.1, 500.0],
            "gt_y_m": [0.0, 0.0, 0.0],
            "gt_speed_mps": [20.0, 22.0, 30.0],
            "space_headway_m": [15.0, 25.0, 5.0],
        }
    )

    matched = match_tracks_to_ground_truth(pred_df, gt_df, max_dist_m=1.0)

    row1 = matched[matched["track_id"] == 1].iloc[0]
    row2 = matched[matched["track_id"] == 2].iloc[0]
    assert row1["matched_vehicle_id"] == 101
    assert row1["gt_speed_mps"] == pytest.approx(20.0)
    assert row2["matched_vehicle_id"] == 102
    assert row2["gt_headway_m"] == pytest.approx(25.0)


def test_match_tracks_to_ground_truth_respects_max_dist():
    pred_df = pd.DataFrame({"frame": [0], "time_s": [0.0], "track_id": [1], "world_x": [0.0], "world_y": [0.0]})
    gt_df = pd.DataFrame(
        {
            "vehicle_id": [101],
            "time_s": [0.0],
            "gt_x_m": [50.0],  # far away
            "gt_y_m": [0.0],
            "gt_speed_mps": [20.0],
            "space_headway_m": [15.0],
        }
    )
    matched = match_tracks_to_ground_truth(pred_df, gt_df, max_dist_m=1.0)
    assert pd.isna(matched.iloc[0]["matched_vehicle_id"])


def test_compute_metrics_reports_mae_mape_and_bias():
    matched_df = pd.DataFrame(
        {
            "speed_mps": [22.0, 18.0],
            "gt_speed_mps": [20.0, 20.0],
            "headway_m": [12.0, 12.0],
            "gt_headway_m": [10.0, 10.0],
            "lane": [0, 1],
            "occluded": [False, True],
        }
    )
    metrics = compute_metrics(matched_df)

    assert metrics["speed"]["n"] == 2
    assert metrics["speed"]["mae"] == pytest.approx(2.0)
    assert metrics["speed"]["bias"] == pytest.approx(0.0)  # errors are +2 and -2
    assert metrics["headway"]["mae"] == pytest.approx(2.0)
    assert metrics["headway"]["bias"] == pytest.approx(2.0)  # consistently over-estimating
    assert "0" in metrics["speed_by_lane"]
    assert "True" in metrics["speed_by_occlusion"]


def test_compute_metrics_handles_no_valid_rows():
    matched_df = pd.DataFrame({"speed_mps": [float("nan")], "gt_speed_mps": [float("nan")]})
    metrics = compute_metrics(matched_df)
    assert metrics["speed"]["n"] == 0
