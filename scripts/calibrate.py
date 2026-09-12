"""Interactive ground-plane calibration: click point correspondences on a
reference video frame and enter each point's real-world (longitudinal,
lateral) coordinates in meters, to fit a homography.

For validating against NGSIM: calibrate directly in NGSIM's own coordinate
frame so `scripts/evaluate.py` can compare apples to apples. To do that,
instead of measuring points yourself, pick >= 4 moments where a specific,
visually identifiable vehicle is at a clear position in the frame, look up
that vehicle's (Local_X, Local_Y) in the ground-truth CSV at the matching
Global_Time (Local_X/Y are in feet — convert to meters, 1 ft = 0.3048 m), and
enter that as the world coordinate for the pixel you click. Use
scripts/estimate_time_offset.py first to know which Global_Time corresponds
to which video frame.

Usage:
    python scripts/calibrate.py data/sb-camera4-0750am-0805am.avi --frame 0 --out results/homography.json

Click at least 4 points spread across the frame (varying both depth and lane
position gives a better-conditioned fit), press any key after each click to
be prompted for that point's real-world coordinates, and press 'q' in the
image window when done.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from roadside_headway.calibration.homography import Homography


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", help="Path to the reference video")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to calibrate on")
    parser.add_argument("--out", default="results/homography.json")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Could not read frame {args.frame} from {args.video}")

    pixel_points: list[tuple[float, float]] = []
    world_points: list[tuple[float, float]] = []
    display = frame.copy()

    def on_click(event, x, y, _flags, _param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        cv2.circle(display, (x, y), 4, (0, 0, 255), -1)
        cv2.imshow("calibrate", display)
        print(f"\nClicked pixel ({x}, {y}).")
        s = float(input("  real-world longitudinal position s (meters): "))
        lateral = float(input("  real-world lateral position l (meters): "))
        pixel_points.append((float(x), float(y)))
        world_points.append((s, lateral))
        print(f"  -> {len(pixel_points)} point(s) recorded.")

    cv2.namedWindow("calibrate")
    cv2.setMouseCallback("calibrate", on_click)
    cv2.imshow("calibrate", display)
    print("Click point correspondences on the image window. Press 'q' in the window when done (need >= 4 points).")
    while cv2.waitKey(50) & 0xFF != ord("q"):
        pass
    cv2.destroyAllWindows()

    if len(pixel_points) < 4:
        raise SystemExit(f"Only {len(pixel_points)} point(s) recorded — need at least 4.")

    homography = Homography.fit(pixel_points, world_points)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    homography.save(args.out)
    print(f"\nSaved homography to {args.out}")
    print(f"Mean reprojection error on calibration points: {homography.reprojection_error_m():.3f} m")
    print("(This is only self-consistency on the points you just gave it, not full validation.)")


if __name__ == "__main__":
    main()
