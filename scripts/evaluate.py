"""Match our pipeline's predicted trajectories to NGSIM ground truth and report
speed/headway error metrics: MAE, MAPE, and signed bias.

Bias matters here specifically: our contact-point-based headway (bottom-center
of the detected box) and NGSIM's front-center-based Space_Headway differ by a
roughly constant offset (~one vehicle length), not just noise — see README.

Usage:
    python scripts/evaluate.py results/trajectories.csv \\
        --gt-csv data/us101_trajectories_0750am-0805am.csv \\
        --video-start-epoch-ms 1118846979300 \\
        --out results/metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from roadside_headway.evaluation.ngsim_gt import compute_metrics, load_ground_truth, match_tracks_to_ground_truth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("predictions_csv", help="Output of scripts/run_pipeline.py")
    parser.add_argument("--gt-csv", default="data/us101_trajectories_0750am-0805am.csv")
    parser.add_argument(
        "--video-start-epoch-ms",
        type=int,
        required=True,
        help="Global_Time (epoch ms) at video frame 0 — see scripts/estimate_time_offset.py",
    )
    parser.add_argument("--max-dist-m", type=float, default=3.0)
    parser.add_argument("--out", default="results/metrics.json")
    args = parser.parse_args()

    pred_df = pd.read_csv(args.predictions_csv)
    gt_df = load_ground_truth(args.gt_csv, args.video_start_epoch_ms)
    matched = match_tracks_to_ground_truth(pred_df, gt_df, max_dist_m=args.max_dist_m)
    metrics = compute_metrics(matched)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    matched_out = out_path.with_name(out_path.stem + "_matched.csv")
    matched.to_csv(matched_out, index=False)
    print(f"\nWrote metrics to {out_path} and matched rows to {matched_out}")


if __name__ == "__main__":
    main()
