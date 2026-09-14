import pandas as pd

from roadside_headway.evaluation.ngsim_camera_coverage import (
    CAMERA_COVERAGE_POLYGONS,
    collect_world_points_by_frame,
    rows_in_camera_view,
)


def test_all_eight_cameras_present():
    assert set(CAMERA_COVERAGE_POLYGONS.keys()) == set(range(1, 9))


def test_rows_in_camera_view_selects_points_inside_polygon():
    df = pd.DataFrame(
        {
            "global_x": [6451900.0, 0.0, 6451950.0],
            "global_y": [1872650.0, 0.0, 1872600.0],
        }
    )
    mask = rows_in_camera_view(df, camera=4)
    assert list(mask) == [True, False, True]


def test_collect_world_points_by_frame(tmp_path):
    csv_path = tmp_path / "gt.csv"
    csv_path.write_text(
        "global_time,local_x,local_y,global_x,global_y\n"
        "1000,10.0,500.0,6451900.0,1872650.0\n"  # inside camera 4, frame 1
        "1000,20.0,600.0,0.0,0.0\n"  # outside camera 4 -- excluded
        "1100,15.0,550.0,6451950.0,1872600.0\n"  # inside camera 4, frame 2
    )
    result = collect_world_points_by_frame(str(csv_path), camera=4, video_start_epoch_ms=900, num_frames=3)

    assert list(result.keys()) == [1, 2]
    assert result[1] == [(500.0 * 0.3048, 10.0 * 0.3048)]
    assert result[2] == [(550.0 * 0.3048, 15.0 * 0.3048)]
