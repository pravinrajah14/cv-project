import pandas as pd
import pytest

from roadside_headway.evaluation.ngsim_gt import load_ground_truth


def test_load_ground_truth_maps_local_y_to_longitudinal_x(tmp_path):
    # Local_Y is NGSIM's longitudinal coordinate and Local_X its lateral one —
    # this must map to gt_x_m=longitudinal, gt_y_m=lateral to match the
    # pipeline's world_x=longitudinal / world_y=lateral convention. A previous
    # version of load_ground_truth had these swapped, which silently made
    # every ground-truth match fail (distances compared longitudinal metres
    # against lateral metres) without erroring.
    csv_path = tmp_path / "gt.csv"
    csv_path.write_text(
        "vehicle_id,global_time,local_x,local_y,v_vel,lane_id,space_headway,time_headway\n"
        "1,1000,10.0,500.0,30.0,2,50.0,1.5\n"
    )
    gt_df = load_ground_truth(str(csv_path), video_start_epoch_ms=0)

    row = gt_df.iloc[0]
    assert row["gt_x_m"] == pytest.approx(500.0 * 0.3048)  # longitudinal (Local_Y)
    assert row["gt_y_m"] == pytest.approx(10.0 * 0.3048)  # lateral (Local_X)
    assert row["time_s"] == pytest.approx(1.0)
    assert row["gt_speed_mps"] == pytest.approx(30.0 * 0.3048)
    assert row["space_headway_m"] == pytest.approx(50.0 * 0.3048)
