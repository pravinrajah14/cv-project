from __future__ import annotations

import numpy as np
import pandas as pd

FEET_TO_METERS = 0.3048


def load_ground_truth(csv_path: str, video_start_epoch_ms: int) -> pd.DataFrame:
    """Load an NGSIM trajectory CSV (already filtered to one site) and convert
    to SI units with a `time_s` column relative to `video_start_epoch_ms` — the
    Global_Time (epoch ms) that corresponds to frame 0 of the matching
    single-camera video. See scripts/estimate_time_offset.py for how to derive
    that anchor for a given site/time-window.
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    return pd.DataFrame(
        {
            "vehicle_id": df["vehicle_id"],
            "time_s": (df["global_time"] - video_start_epoch_ms) / 1000.0,
            "gt_x_m": df["local_x"] * FEET_TO_METERS,
            "gt_y_m": df["local_y"] * FEET_TO_METERS,
            "gt_speed_mps": df["v_vel"] * FEET_TO_METERS,
            "lane_id": df["lane_id"],
            "space_headway_m": df["space_headway"] * FEET_TO_METERS,
            "time_headway_s": df["time_headway"],
        }
    )


def match_tracks_to_ground_truth(
    pred_df: pd.DataFrame, gt_df: pd.DataFrame, max_dist_m: float = 3.0, time_round: int = 1
) -> pd.DataFrame:
    """Attach the nearest-in-time-and-space NGSIM ground-truth vehicle to each
    predicted (frame, track_id) row.

    Assumes the homography used to produce `pred_df` was calibrated in the same
    world coordinate frame as the ground truth (e.g. by using known GT vehicle
    positions at known times as calibration correspondences — see README).

    NOTE: our own longitudinal position is the vehicle's *contact point*
    (bottom-center of the detected box), while NGSIM's Local_Y / Space_Headway
    are measured at the vehicle's *front-center*. Expect a roughly constant
    offset (on the order of one vehicle length) between our raw headway and
    NGSIM's — that's why `compute_metrics` reports signed bias, not just MAE.
    """
    if pred_df.empty:
        return pred_df.assign(matched_vehicle_id=pd.NA, gt_speed_mps=np.nan, gt_headway_m=np.nan)

    pred = pred_df.copy()
    gt = gt_df.copy()
    pred["time_key"] = pred["time_s"].round(time_round)
    gt["time_key"] = gt["time_s"].round(time_round)

    merged = pred.merge(gt, on="time_key", suffixes=("", "_gt"))
    merged["dist_m"] = np.hypot(merged["world_x"] - merged["gt_x_m"], merged["world_y"] - merged["gt_y_m"])
    merged = merged[merged["dist_m"] <= max_dist_m]
    merged = merged.sort_values("dist_m").drop_duplicates(subset=["frame", "track_id"], keep="first")

    result = pred_df.merge(
        merged[["frame", "track_id", "vehicle_id", "gt_speed_mps", "space_headway_m", "dist_m"]],
        on=["frame", "track_id"],
        how="left",
    ).rename(columns={"vehicle_id": "matched_vehicle_id", "space_headway_m": "gt_headway_m"})
    return result


def _mae_mape_bias(pred: pd.Series, gt: pd.Series) -> dict:
    pred_arr, gt_arr = np.asarray(pred, dtype=float), np.asarray(gt, dtype=float)
    mask = ~np.isnan(pred_arr) & ~np.isnan(gt_arr) & (gt_arr != 0)
    if mask.sum() == 0:
        return {"n": 0, "mae": float("nan"), "mape_pct": float("nan"), "bias": float("nan")}
    err = pred_arr[mask] - gt_arr[mask]
    return {
        "n": int(mask.sum()),
        "mae": float(np.mean(np.abs(err))),
        "mape_pct": float(np.mean(np.abs(err) / np.abs(gt_arr[mask])) * 100),
        "bias": float(np.mean(err)),
    }


def compute_metrics(matched_df: pd.DataFrame) -> dict:
    """MAE, MAPE (%), and signed bias for speed and headway against matched
    ground truth, overall and broken out by lane and by occlusion flag."""
    metrics = {
        "speed": _mae_mape_bias(matched_df.get("speed_mps", pd.Series(dtype=float)), matched_df.get("gt_speed_mps", pd.Series(dtype=float))),
        "headway": _mae_mape_bias(matched_df.get("headway_m", pd.Series(dtype=float)), matched_df.get("gt_headway_m", pd.Series(dtype=float))),
    }
    if "lane" in matched_df.columns:
        metrics["speed_by_lane"] = {
            str(lane): _mae_mape_bias(g["speed_mps"], g["gt_speed_mps"])
            for lane, g in matched_df.groupby("lane")
            if not pd.isna(lane)
        }
        metrics["headway_by_lane"] = {
            str(lane): _mae_mape_bias(g["headway_m"], g["gt_headway_m"])
            for lane, g in matched_df.groupby("lane")
            if not pd.isna(lane)
        }
    if "occluded" in matched_df.columns:
        metrics["speed_by_occlusion"] = {
            str(occ): _mae_mape_bias(g["speed_mps"], g["gt_speed_mps"]) for occ, g in matched_df.groupby("occluded")
        }
    return metrics
