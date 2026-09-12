from __future__ import annotations

import pandas as pd

from roadside_headway.calibration.lanes import LaneBoundaries
from roadside_headway.headway.trajectory import compute_speed, smooth_positions


def assign_lanes(df: pd.DataFrame, lanes: LaneBoundaries, lateral_col: str = "world_y") -> pd.DataFrame:
    df = df.copy()
    df["lane"] = df[lateral_col].apply(lanes.lane_of)
    return df


def compute_headway(df: pd.DataFrame, longitudinal_col: str = "world_x", direction: int = 1) -> pd.DataFrame:
    """For each frame and lane, compute headway (m) to the vehicle immediately ahead.

    `direction` says which sign of `longitudinal_col` corresponds to the
    direction of travel (+1 or -1); "progress" = direction * position always
    increases in the direction of travel, so sorting by it and diffing gives
    the correct positive gap regardless of the world-frame's sign convention.
    """
    df = df.copy()
    df["progress"] = direction * df[longitudinal_col]
    df["headway_m"] = float("nan")
    df["lead_track_id"] = pd.NA

    for (_frame, lane), group in df.groupby(["frame", "lane"]):
        if pd.isna(lane) or len(group) < 2:
            continue
        ordered = group.sort_values("progress")
        idx = ordered.index.to_numpy()
        progress = ordered["progress"].to_numpy()
        track_ids = ordered["track_id"].to_numpy()
        gaps = progress[1:] - progress[:-1]
        df.loc[idx[:-1], "headway_m"] = gaps
        df.loc[idx[:-1], "lead_track_id"] = track_ids[1:]

    return df.drop(columns=["progress"])


def compute_speed_per_track(
    df: pd.DataFrame,
    time_col: str = "time_s",
    longitudinal_col: str = "world_x",
    lateral_col: str = "world_y",
    smoothing_window: int = 5,
) -> pd.DataFrame:
    df = df.sort_values(["track_id", time_col]).copy()
    df["speed_mps"] = float("nan")
    for track_id, group in df.groupby("track_id"):
        if len(group) < 2:
            continue
        positions = group[[longitudinal_col, lateral_col]].to_numpy()
        window = min(smoothing_window, len(positions))
        smoothed = smooth_positions(positions, window=window)
        times = group[time_col].to_numpy()
        speed = compute_speed(times, smoothed[:, 0])
        df.loc[group.index, "speed_mps"] = speed
    return df
