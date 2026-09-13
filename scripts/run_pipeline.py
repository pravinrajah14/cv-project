"""Run detection + tracking + (optional) depth-assisted occlusion handling +
homography projection over a video, and write the resulting per-frame,
per-track trajectory table (with per-lane headway and per-track speed already
computed).

Usage:
    python scripts/run_pipeline.py data/sb-camera4-0750am-0805am.avi \\
        --homography results/homography.json --lanes results/lanes.json \\
        --out results/trajectories.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from roadside_headway.calibration.homography import Homography
from roadside_headway.calibration.lanes import LaneBoundaries
from roadside_headway.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video")
    parser.add_argument("--homography", default="results/homography.json")
    parser.add_argument("--lanes", default="results/lanes.json")
    parser.add_argument("--direction", type=int, default=1, choices=[-1, 1])
    parser.add_argument(
        "--no-depth", action="store_true", help="Skip the depth model (faster; disables occlusion handling)"
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--region-margin-frac",
        type=float,
        default=1.0,
        help="Drop detections whose projected position falls outside the calibrated "
        "region expanded by this fraction of its span (guards against homography "
        "extrapolation blowup for e.g. vehicles on an adjacent ramp)",
    )
    parser.add_argument("--out", default="results/trajectories.csv")
    args = parser.parse_args()

    homography = Homography.load(args.homography)
    lanes = LaneBoundaries.load(args.lanes)

    df = run_pipeline(
        args.video,
        homography=homography,
        lanes=lanes,
        direction=args.direction,
        use_depth=not args.no_depth,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        device=args.device,
        region_margin_frac=args.region_margin_frac,
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    n_tracks = df["track_id"].nunique() if not df.empty else 0
    print(f"Wrote {len(df)} rows ({n_tracks} tracks) to {args.out}")


if __name__ == "__main__":
    main()
