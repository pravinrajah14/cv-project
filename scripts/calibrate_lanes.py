"""Define lane lateral boundaries by clicking one point per lane-boundary line
on a reference frame (already calibrated with scripts/calibrate.py) and
projecting through the homography to get each boundary's lateral coordinate.

Usage:
    python scripts/calibrate_lanes.py data/sb-camera4-0750am-0805am.avi \\
        --homography results/homography.json --frame 0 --out results/lanes.json

Click points along each lane-boundary line, in order across the road (left to
right or right to left, as long as it's consistent). N clicks defines N-1
lanes. Press 'q' when done.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from roadside_headway.calibration.homography import Homography
from roadside_headway.calibration.lanes import LaneBoundaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video")
    parser.add_argument("--homography", default="results/homography.json")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--out", default="results/lanes.json")
    args = parser.parse_args()

    homography = Homography.load(args.homography)

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Could not read frame {args.frame} from {args.video}")

    boundaries: list[float] = []
    display = frame.copy()

    def on_click(event, x, y, _flags, _param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        cv2.circle(display, (x, y), 4, (0, 255, 0), -1)
        cv2.imshow("calibrate_lanes", display)
        _s, lateral = homography.pixel_to_world((x, y))
        boundaries.append(lateral)
        print(f"Clicked pixel ({x}, {y}) -> lateral = {lateral:.2f} m ({len(boundaries)} boundary point(s) so far)")

    cv2.namedWindow("calibrate_lanes")
    cv2.setMouseCallback("calibrate_lanes", on_click)
    cv2.imshow("calibrate_lanes", display)
    print("Click one point per lane boundary line. Press 'q' when done (need >= 2 points for 1 lane).")
    while cv2.waitKey(50) & 0xFF != ord("q"):
        pass
    cv2.destroyAllWindows()

    if len(boundaries) < 2:
        raise SystemExit(f"Only {len(boundaries)} boundary point(s) recorded — need at least 2 (for 1 lane).")

    lanes = LaneBoundaries(boundaries=sorted(boundaries))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    lanes.save(args.out)
    print(f"\nSaved {len(boundaries) - 1} lane(s) to {args.out}")


if __name__ == "__main__":
    main()
