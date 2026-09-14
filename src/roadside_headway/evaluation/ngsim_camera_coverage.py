"""Per-camera field-of-view polygons for the NGSIM US-101 site, in California
State Plane Zone 5 (NAD83, US survey feet) — the same coordinate system as the
ground-truth CSV's `Global_X`/`Global_Y` columns.

Extracted from the official `camera-coverage.shp` shapefile (US-101 GIS files
bundle, from the NGSIM US-101 metadata documentation hub, view id 98ir-zwui,
attachment "US-101-LosAngeles-CA.zip" -> us-101-gis-files.zip). Each polygon
is the camera's coverage quadrilateral; camera IDs match the video filenames
(`sb-camera<N>-...avi`).
"""

from __future__ import annotations

FEET_TO_METERS = 0.3048

CAMERA_COVERAGE_POLYGONS: dict[int, list[tuple[float, float]]] = {
    1: [
        (6452518.52083978, 1872117.863341065),
        (6452771.406103, 1871919.0888983593),
        (6452729.442609539, 1871852.830750791),
        (6452471.035834023, 1872060.439613172),
    ],
    2: [
        (6452282.200113452, 1872324.3679009867),
        (6452519.625142239, 1872118.9676435243),
        (6452471.035834023, 1872059.3353107127),
        (6452232.5065027755, 1872262.526963256),
    ],
    3: [
        (6452041.314936957, 1872534.6454943118),
        (6452282.568214271, 1872324.3494959457),
        (6452234.531057284, 1872262.434938051),
        (6451991.142795215, 1872472.7309364174),
    ],
    4: [
        (6451841.6938623665, 1872708.6467518434),
        (6452041.314936957, 1872534.6454943118),
        (6451992.210287593, 1872472.7309364174),
        (6451791.5217206245, 1872648.8671787037),
    ],
    5: [
        (6451698.355403127, 1872838.034190012),
        (6451843.239885811, 1872709.0516627452),
        (6451792.883693659, 1872648.9776089496),
        (6451644.4654431045, 1872775.3098103136),
    ],
    6: [
        (6451517.249799773, 1872993.5199763062),
        (6451698.355403127, 1872838.034190012),
        (6451644.4654431045, 1872775.3098103136),
        (6451468.660491556, 1872928.1452707052),
    ],
    7: [
        (6451315.746611187, 1873183.1636194733),
        (6451518.468572609, 1872993.7946016176),
        (6451467.484606263, 1872929.457691705),
        (6451262.33483692, 1873126.1101333245),
    ],
    8: [
        (6451034.1208923245, 1873376.1743492107),
        (6451096.0299943155, 1873427.1583155566),
        (6451315.746611187, 1873181.9497155126),
        (6451263.548740881, 1873126.1101333245),
        (6451035.334796285, 1873374.9604452502),
    ],
}


def rows_in_camera_view(df, camera: int, global_x_col: str = "global_x", global_y_col: str = "global_y"):
    """Boolean mask selecting ground-truth rows that fall inside a camera's
    coverage polygon. `df` is a pandas DataFrame with Global_X/Global_Y columns
    (feet, State Plane Zone 5 NAD83)."""
    from matplotlib.path import Path

    polygon = Path(CAMERA_COVERAGE_POLYGONS[camera])
    points = df[[global_x_col, global_y_col]].to_numpy()
    return polygon.contains_points(points)


def collect_world_points_by_frame(
    gt_csv: str, camera: int, video_start_epoch_ms: int, num_frames: int
) -> dict[int, list[tuple[float, float]]]:
    """Ground-truth vehicle world positions (meters; world_x=longitudinal,
    world_y=lateral) visible in a camera's coverage, keyed by video frame
    index — for matching against our own detections at the same instant."""
    import pandas as pd

    df = pd.read_csv(gt_csv)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df[rows_in_camera_view(df, camera)]

    frames_world_points: dict[int, list[tuple[float, float]]] = {}
    for frame_idx in range(num_frames):
        global_time = video_start_epoch_ms + frame_idx * 100
        rows = df[df["global_time"] == global_time]
        points = list(zip(rows["local_y"] * FEET_TO_METERS, rows["local_x"] * FEET_TO_METERS))
        if points:
            frames_world_points[frame_idx] = points
    return frames_world_points
