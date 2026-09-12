"""Download a single-camera NGSIM US-101 video clip and the matching
trajectory ground-truth CSV (filtered to that one 15-minute time window) via
the Socrata SODA API on data.transportation.gov.

Endpoints were verified manually (curl) before writing this script:
  - Trajectory CSV: dataset 8ect-6jqj ("NGSIM Vehicle Trajectories and
    Supporting Data"), filtered by `location` + a `global_time` range, via
    https://data.transportation.gov/resource/8ect-6jqj.csv
  - Raw video attachments for the US-101 videos dataset (view id 4qzi-thur),
    listed via https://data.transportation.gov/api/views/4qzi-thur.json and
    fetched via https://data.transportation.gov/api/views/4qzi-thur/files/<assetId>

The full us-101 ground-truth table is ~4.8M rows across all three 15-minute
periods (much bigger than a single-segment v1 needs), so this filters by a
global_time range up front rather than downloading everything and slicing
locally.

Usage:
    python scripts/download_ngsim.py --camera 4 --time-window 0750am-0805am

NOTE: after downloading, open the video and confirm the field of view is a
straight, unobstructed mainline segment (this site includes an on-ramp merge,
which is out of scope for v1 — see README limitations). Try a different
--camera if the default view isn't usable.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import requests

from roadside_headway.evaluation.ngsim_sites import TIME_WINDOWS, nominal_epoch_range_ms

TRAJECTORY_RESOURCE = "8ect-6jqj"
US101_VIDEO_VIEW = "4qzi-thur"
SOCRATA_BASE = "https://data.transportation.gov"


def _get_with_retries(url: str, params: dict | None = None, timeout: int = 180, retries: int = 4) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as exc:  # timeouts, connection errors, 5xx, etc.
            last_error = exc
            wait_s = 2**attempt
            print(f"  request failed ({exc.__class__.__name__}: {exc}); retrying in {wait_s}s...")
            time.sleep(wait_s)
    raise RuntimeError(f"Giving up after {retries} attempts on {url}") from last_error


def fetch_trajectory_csv(
    out_path: Path, location: str, time_window: str, page_size: int = 25000, buffer_s: int = 90
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    start_ms, end_ms = nominal_epoch_range_ms(location, time_window, buffer_s=buffer_s)
    where_clause = f"location='{location}' AND global_time between {start_ms} and {end_ms}"

    offset = 0
    header_written = False
    with tmp_path.open("w", newline="") as f:
        while True:
            params = {
                "$where": where_clause,
                "$order": "global_time, vehicle_id",  # stable ordering — required for correct offset pagination
                "$limit": page_size,
                "$offset": offset,
            }
            resp = _get_with_retries(f"{SOCRATA_BASE}/resource/{TRAJECTORY_RESOURCE}.csv", params=params)
            lines = resp.text.splitlines()
            n_rows = max(len(lines) - 1, 0)
            if n_rows == 0:
                break
            if not header_written:
                f.write(resp.text + "\n")
                header_written = True
            else:
                f.write("\n".join(lines[1:]) + "\n")
            offset += page_size
            print(f"  fetched {n_rows} rows (offset now {offset})")
            if n_rows < page_size:
                break
    tmp_path.replace(out_path)


def list_video_attachments() -> list[dict]:
    resp = requests.get(f"{SOCRATA_BASE}/api/views/{US101_VIDEO_VIEW}.json", timeout=60)
    resp.raise_for_status()
    return resp.json().get("attachments", [])


def fetch_video(filename: str, out_path: Path) -> None:
    attachments = list_video_attachments()
    match = next((a for a in attachments if a["filename"] == filename), None)
    if match is None:
        available = "\n".join(sorted(a["filename"] for a in attachments))
        raise ValueError(f"'{filename}' not found in the US-101 video attachments. Available:\n{available}")

    url = f"{SOCRATA_BASE}/api/views/{US101_VIDEO_VIEW}/files/{match['assetId']}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    with requests.get(url, params={"download": "true", "filename": filename}, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0))
        written = 0
        with tmp_path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                written += len(chunk)
                if total:
                    print(f"\r  {written / 1e6:.1f} / {total / 1e6:.1f} MB", end="")
    print()
    tmp_path.replace(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=4, help="Camera number 1-8 (default: 4, roughly central)")
    parser.add_argument("--time-window", default=TIME_WINDOWS[0], choices=TIME_WINDOWS)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--skip-video", action="store_true", help="Only fetch the trajectory ground-truth CSV")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    csv_path = data_dir / f"us101_trajectories_{args.time_window}.csv"
    if csv_path.exists():
        print(f"{csv_path} already exists, skipping.")
    else:
        print(f"Downloading US-101 trajectory ground truth ({args.time_window}) to {csv_path} ...")
        fetch_trajectory_csv(csv_path, location="us-101", time_window=args.time_window)

    if not args.skip_video:
        filename = f"sb-camera{args.camera}-{args.time_window}.avi"
        video_path = data_dir / filename
        if video_path.exists():
            print(f"{video_path} already exists, skipping.")
        else:
            print(f"Downloading {filename} ...")
            fetch_video(filename, video_path)

    print(
        "\nDone. Next: open the video and confirm it's a straight, unobstructed "
        "mainline segment before running scripts/calibrate.py."
    )


if __name__ == "__main__":
    main()
