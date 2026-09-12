"""Determine the Global_Time (epoch ms) that corresponds to frame 0 of a given
NGSIM single-camera video, by finding the earliest Global_Time in the ground
truth trajectory data whose value falls near that time-window's nominal local
start time. All 8 cameras at a site are documented as synchronized, so this
anchor is shared across camera numbers for the same site/time-window.

This is an estimate derived from the ground truth's own timestamps, not an
independently verified sync signal against the actual video frames (that
requires eyeballing a downloaded clip, which hasn't been done here) —
sanity-check it once by confirming a distinctive vehicle's on-screen
appearance time roughly matches its Global_Time in the CSV, per the README.

Usage:
    python scripts/estimate_time_offset.py --location us-101 --time-window 0750am-0805am
"""

from __future__ import annotations

import argparse

import requests

from roadside_headway.evaluation.ngsim_sites import SITE_DATES, TIME_WINDOWS, nominal_epoch_range_ms

SOCRATA_BASE = "https://data.transportation.gov"
TRAJECTORY_RESOURCE = "8ect-6jqj"


def find_video_start_epoch_ms(location: str, time_window: str, buffer_s: int = 90) -> int:
    lower_bound, _upper_bound = nominal_epoch_range_ms(location, time_window, buffer_s=buffer_s)
    resp = requests.get(
        f"{SOCRATA_BASE}/resource/{TRAJECTORY_RESOURCE}.json",
        params={
            "$select": "min(global_time) as min_gt",
            "location": location,
            "$where": f"global_time>={lower_bound}",
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    if not data or data[0].get("min_gt") is None:
        raise ValueError(f"No trajectory rows found for location={location} near {time_window}")
    return int(data[0]["min_gt"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--location", default="us-101", choices=list(SITE_DATES))
    parser.add_argument("--time-window", default=TIME_WINDOWS[0], choices=TIME_WINDOWS)
    args = parser.parse_args()

    epoch_ms = find_video_start_epoch_ms(args.location, args.time_window)
    print(f"Estimated Global_Time at video frame 0: {epoch_ms}")
    print("Pass this to scripts/evaluate.py as --video-start-epoch-ms")


if __name__ == "__main__":
    main()
