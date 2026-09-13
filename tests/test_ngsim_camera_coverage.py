import pandas as pd

from roadside_headway.evaluation.ngsim_camera_coverage import CAMERA_COVERAGE_POLYGONS, rows_in_camera_view


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
