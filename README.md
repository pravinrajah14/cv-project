# Roadside Vehicle Headway Estimation

Estimate real-world vehicle **headway** (following distance to the vehicle
ahead in the same lane) and per-vehicle **speed** from a single fixed,
elevated/oblique roadside camera — no LiDAR, no 3D annotations, no stereo.

This is a variant of an earlier idea (2D detector + monocular depth →
pseudo-3D boxes for a *vehicle-mounted* camera, validated on nuScenes-mini).
For a **fixed** camera, the geometry is different in one important way: since
the camera never moves, real-world position along the road is recovered from
a **ground-plane homography** applied to each vehicle's road-contact point —
not from monocular depth. Depth is kept in the pipeline, but only as an
**auxiliary** signal (see [Depth's role](#depths-role-auxiliary-not-primary)
below), which is a deliberate departure from the vehicle-mounted version of
this idea.

## Pipeline

```
video frame
   │
   ├─► YOLO11s detector (car/truck/bus/motorcycle)
   │        │
   │        ▼
   ├─► ByteTrack (Ultralytics built-in) ──► per-vehicle track IDs across frames
   │        │
   ├─► Depth Anything V2 (Small) ──► per-frame relative depth map
   │        │                              │
   │        ▼                              ▼
   │   occlusion check (tracking/occlusion.py): for overlapping boxes, the
   │   optically nearer one occludes the farther one — flag the farther
   │   vehicle's contact point as unreliable this frame
   │        │
   ▼        ▼
contact point (bbox bottom-center), or constant-velocity extrapolation
from track history if flagged occluded
   │
   ▼
ground-plane homography (calibration/homography.py) ──► real-world (s, l) in meters
   │
   ▼
lane assignment (calibration/lanes.py) + per-lane ordering
   │
   ▼
headway (headway/compute.py: gap to the vehicle ahead in the same lane)
speed (headway/trajectory.py: smoothed finite difference of position)
```

`src/roadside_headway/pipeline.py` runs all of this end to end for one video.

### Depth's role: auxiliary, not primary

For a vehicle-mounted camera, depth is the main way to get distance. For a
**fixed** camera looking down at a known, (approximately) planar road, a
calibrated homography on the vehicle's ground-contact pixel is already an
accurate and much simpler way to get real-world position — that's the
standard approach in the traffic-camera speed-measurement literature (e.g.
BrnoCompSpeed). So here, Depth Anything V2 is used only to figure out **which
of two overlapping vehicles is physically nearer the camera lens**
(`tracking/occlusion.py`), so the farther one's occluded/clipped bounding-box
bottom edge can be replaced with a constant-velocity extrapolation from its
own track history instead of a corrupted homography projection. This is
qualitatively useful for occlusion handling but is **not separately
validated** — see [Limitations](#limitations).

## Datasets

**Primary — NGSIM, US-101 (Hollywood Freeway), single camera, one segment.**
Free, no registration wall for the trajectory data or the video (verified
directly against the live Socrata endpoints — see `scripts/download_ngsim.py`
and `scripts/estimate_time_offset.py`). The trajectory CSV already contains,
per vehicle per 0.1 s frame: position, instantaneous speed, lane, and —
critically — `Space_Headway` and `Time_Headway`: real ground truth for exactly
what this project estimates.

- Site: southbound US-101, Los Angeles, 2005-06-15, three 15-minute periods
  (`0750am-0805am`, `0805am-0820am`, `0820am-0835am`), 8 synchronized cameras.
- v1 uses **one camera, one segment** (default: camera 4, `0750am-0805am`) —
  not all 8 cameras, not the full 45 minutes.
- **Known GT caveat**: NGSIM's own trajectory reconstruction has documented
  noise/smoothing artifacts (see the follow-up literature on re-processing
  NGSIM trajectories). Reported errors below include this noise floor — they
  are not purely "our pipeline's error."
- **Definition mismatch to expect**: NGSIM's `Space_Headway` is measured
  front-center-to-front-center; this pipeline's headway is measured
  contact-point-to-contact-point (bbox bottom-center). Expect a roughly
  constant offset on the order of one vehicle length — that's why
  `evaluate.py` reports signed **bias**, not just MAE/MAPE.

**Stretch/optional — BrnoCompSpeed.** Purpose-built for monocular speed
validation (LIDAR light-barrier ground truth, published camera calibration).
Access requires emailing the dataset author — not wired into this repo yet.

UA-DETRAC was considered and ruled out: no speed/position ground truth.

### Getting the data

```bash
python scripts/download_ngsim.py --camera 4 --time-window 0750am-0805am
```

Downloads `data/us101_trajectories_0750am-0805am.csv` (ground truth for that
one 15-minute window — the full US-101 table is ~4.8M rows across all three
periods, so this filters server-side rather than pulling everything) and
`data/sb-camera4-0750am-0805am.avi` (~380 MB). **Open the video and confirm
it's a straight, unobstructed mainline segment** — the US-101 study area
includes an on-ramp/off-ramp merge, which is out of scope for v1 (see
below) — before calibrating. Try a different `--camera` if the default view
includes the merge.

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Requires Python ≥3.10. Tested on an Apple M3 (16 GB) with the `mps` PyTorch
backend.

**License note**: Ultralytics (YOLO11) is AGPL-3.0-licensed. Fine for a
research/portfolio repo; worth knowing if you ever want to relicense this
project permissively.

## Usage

Two calibration paths exist. Use the ground-truth-anchored one when validating
against NGSIM (it's what produced the numbers in [Results](#results) below);
use the manual one for footage with no ground truth available.

```bash
# 1. Download data
python scripts/download_ngsim.py --camera 4 --time-window 0750am-0805am

# 2a. Calibrate automatically against ground truth (recommended for NGSIM):
#     tracks our own detector/tracker for --num-frames, matches its pixel
#     positions against real vehicle positions at the same instants via
#     iterative nearest-neighbor fitting, and resolves the direction-of-travel
#     ambiguity (see Limitations) using tracked motion. Needs a
#     camera-coverage polygon for the chosen camera (US-101 only, all 8
#     cameras, in evaluation/ngsim_camera_coverage.py).
python scripts/estimate_time_offset.py --location us-101 --time-window 0750am-0805am
python scripts/calibrate_from_gt.py data/sb-camera4-0750am-0805am.avi \
    --camera 4 --gt-csv data/us101_trajectories_0750am-0805am.csv \
    --video-start-epoch-ms <value from estimate_time_offset.py> --num-frames 900 \
    --out results/homography.json

# 2b. ...or calibrate manually (interactive; for footage without ground truth).
#     To still validate against NGSIM this way, calibrate directly in NGSIM's
#     coordinate frame — see the docstring in scripts/calibrate.py.
python scripts/calibrate.py data/sb-camera4-0750am-0805am.avi --out results/homography.json

# 3. Define lane boundaries. With ground truth, derive them directly from
#    real per-lane lateral positions:
python scripts/derive_lanes_from_gt.py --camera 4 \
    --gt-csv data/us101_trajectories_0750am-0805am.csv --out results/lanes.json

#    ...or, without ground truth, click them on a reference frame:
python scripts/calibrate_lanes.py data/sb-camera4-0750am-0805am.avi \
    --homography results/homography.json --out results/lanes.json

# 4. Run the full pipeline — on a frame range held out from calibration
#    fitting, so the reported accuracy isn't inflated by evaluating on the
#    same data the homography was fit to. Omit --max-frames to run to the
#    end of the clip (what produced the Results numbers below); add it back
#    for a quicker spot-check.
python scripts/run_pipeline.py data/sb-camera4-0750am-0805am.avi \
    --homography results/homography.json --lanes results/lanes.json \
    --start-frame 900 \
    --out results/trajectories.csv

# 5. Evaluate against NGSIM ground truth
python scripts/evaluate.py results/trajectories.csv \
    --gt-csv data/us101_trajectories_0750am-0805am.csv \
    --video-start-epoch-ms <value from step 2a> \
    --out results/metrics.json
```

## Validation methodology

`scripts/evaluate.py` matches each predicted (frame, track) to the nearest
ground-truth vehicle at the same instant (within a configurable distance
threshold) and reports, for both speed and headway:

- **MAE** — mean absolute error
- **MAPE** — mean absolute percentage error
- **bias** — signed mean error (surfaces systematic offsets, like the
  contact-point-vs-front-center definition mismatch above, instead of hiding
  them inside a symmetric-error metric)

broken out overall, by lane, and by whether the depth-based occlusion flag
was set. Numbers land in `results/metrics.json` — not just plots.

## Results

Two independently calibrated cameras, same site/window (US-101,
`0750am-0805am`). Each homography was fit on frames 0–899 and evaluated on
the held-out remainder of the clip — frames 900–9550, ~14.25 minutes the fit
never saw. Full numbers in `results/metrics_full.json` (camera 4) and
`results/metrics_cam2_full.json` (camera 2).

| Camera | Metric | MAE | MAPE | Bias | n |
|---|---|---|---|---|---|
| 4 | Speed | 3.58 m/s (8.0 mph) | 36.6% | −1.31 m/s | 4785 |
| 4 | Headway | 12.52 m | 54.7% | −6.99 m | 1644 |
| 2 | Speed | 4.68 m/s (10.5 mph) | 33.2% | −3.91 m/s | 9316 |
| 2 | Headway | 20.04 m | 66.0% | −18.42 m | 6195 |

Both cameras land in the same ballpark (speed MAE 3.5–4.7 m/s, headway MAE
12.5–20 m, consistently negative bias — expected, see the headway-definition
mismatch above), which is a reasonable generalization signal. Camera 2's
headway error is meaningfully worse, and its per-lane breakdown
(`results/metrics_cam2_full.json`) has one lane with only 90 matched points
and a much larger MAE than the rest — a rough edge worth noting, not
explained away, in a scene that's visibly busier (an adjacent ramp/plaza
generates extra detections; see below).

Take these as real data points, not a general accuracy claim — see
[Limitations](#limitations) for why: NGSIM's own ground truth carries
reconstruction noise, and this covers one site, one time window, two cameras.
**Four real bugs were caught only by actually running this evaluation twice**,
against real ground truth and a second camera — not by code review:

1. A ground-truth axis mix-up (`gt_x_m`/`gt_y_m` swapped) that silently
   zeroed out every match (n=0 everywhere, no error).
2. A direction-of-travel sign ambiguity inherent to fitting a homography from
   position snapshots alone (speeds came out uniformly negative where ground
   truth was positive) — position-only fitting can't tell which way is
   "forward."
3. A video downloader with no retry logic that died mid-download at 338/380MB
   with no way to resume.
4. **Found only by adding camera 2**: a plain least-squares homography fit
   can have a near-singular matrix (det ≈ −3.6×10⁻⁴) — reasonable residual on
   its own training points, but catastrophically wrong (positions off by tens
   of thousands of meters) for any pixel outside them — when a handful of
   wrong nearest-neighbor correspondences survive distance thresholding.
   Camera 4's fit happened not to trigger this; camera 2's did. Fixed with a
   RANSAC refit plus an explicit sanity bound
   (`Homography.is_within_calibrated_region`) that drops predictions far
   outside the calibrated region regardless of cause.

All four are in the git history. The pattern across all of them: every one
was invisible from reading the code, and each surfaced only by running the
full pipeline against real data — first against ground truth at all, then
against a second camera. That's the argument for treating this run's own
numbers, and any calibration/validation code, with some ongoing skepticism
rather than as settled once a first result looks plausible.

**A fifth issue, not in the code at all**: after fixing bug 4, camera 4's
full-clip re-run was launched *concurrently* with camera 2's — on a 16GB
machine, one of the two full detection+depth+tracking pipelines got silently
OOM-killed. No error, exit code 0, just a missing "wrote N rows" line and a
resource-tracker warning, with the old output file left untouched. That old
(pre-fix) file was then evaluated and reported as "confirmed unchanged" —
plausible-looking, wrong, and only caught by noticing the exit-code-0 process
had no actual success message. Corrected numbers are the ones above (camera
4's did move slightly: MAE 3.92→3.58 m/s speed, 13.47→12.52 m headway).
Moral: run one heavy pipeline job at a time on this hardware, and check logs
for the actual completion message, not just the exit code.

## Out of scope for v1

- Night / adverse-weather footage (daytime, clear-weather clip only)
- Multi-camera stitching or hand-off (single camera field of view only)
- Real-time/streaming performance (offline batch processing)
- Non-straight road geometry — merges, curves, ramps, intersections (straight
  through-traffic segment only; the default US-101 view must be checked for this)
- Long/heavy-occlusion re-identification beyond ByteTrack's IoU association
- Using depth for absolute distance (auxiliary occlusion signal only, per above)

## Limitations

Read before trusting any number this pipeline produces:

- **"Off-the-shelf" needed real tuning to work on this camera angle.**
  Stock `yolo11n` (nano), COCO-pretrained, essentially detects *nothing* on
  NGSIM's overhead camera view — confirmed directly: at confidence 0.05 it
  found zero vehicle boxes on a real frame with ~15 visible vehicles, only
  spurious `person`/`boat` guesses. `yolo11s` (small) at a lower confidence
  threshold (0.1) does detect vehicles reasonably. This isn't a fine-tuned
  model — both are stock COCO weights — but it's a real reminder that "2D
  detector + monocular depth, off-the-shelf" doesn't mean *any* off-the-shelf
  checkpoint works unmodified on an unusual camera angle; the working
  configuration was found empirically against this dataset (see
  `detection/detector.py`) and may need re-tuning for different footage.
- **NGSIM ground truth itself is noisy.** Reported errors are a floor, not a
  clean measure of pipeline error alone.
- **Planar-homography assumption.** Any road grade or calibration-point error
  biases longitudinal position, worse further from the camera (foreshortening).
- **Contact-point proxy.** Bottom-center-of-bbox approximates a vehicle's
  ground-contact point, but varies with vehicle type, pose, and viewing angle;
  only partially corrected by the depth-based occlusion heuristic.
- **Headway definition mismatch with NGSIM** (see above) — expect a
  systematic bias, not pure noise, when comparing to `Space_Headway`.
- **Depth's occlusion-handling benefit is not independently validated** — it's
  a plausible heuristic, exercised in the pipeline, but not measured in
  isolation against a ground truth for "was this the right occluder."
- **Speed uses only the longitudinal velocity component** — a simplification
  that under-represents speed during a lane change.
- **Two-camera, single-site validation.** Results describe US-101 during one
  15-minute window on two of its eight cameras — a real generalization check,
  not a claim about accuracy across sites, weather, or congestion levels.
- **Auto-calibration's direction check covers longitudinal only.**
  `calibrate_from_gt.py` resolves the direction-of-travel mirror ambiguity
  (see Results) using tracked motion, but doesn't run the equivalent check on
  the lateral axis — a lateral mirror is left undetected if the ICP fit
  happens to converge to one. It didn't in either run (per-lane metrics came
  out sane on both cameras), but that's an empirical observation, not a
  guarantee for other camera views.
- **The region-bound safeguard is a heuristic, not a real field-of-view
  mask.** `Homography.is_within_calibrated_region` rejects predictions far
  outside the calibrated points' bounding box (expanded by a margin) —
  it caught camera 2's extrapolation blowup, but a bad prediction that
  happens to land inside the box (rather than thousands of meters away)
  would sail through undetected. Camera 2's frame also includes a visually
  adjacent ramp/plaza area whose vehicles aren't in the tracked ground truth
  at all — its extra false-positive-prone detections were a direct
  contributor to that camera's near-singular homography fit.
- **A calibration-accuracy experiment that didn't pan out, kept in the code
  anyway.** `Detection.leading_edge_point` (bbox leading edge at vertical
  center, picked via `estimate_pixel_direction_sign`) was added on the
  reasoning that it's a closer geometric match to NGSIM's front-center
  definition than `contact_point` (bottom-center) for this near-overhead
  camera. Tested on camera 4: fitting residual improved slightly
  (2.73→2.66 m) and headway MAE improved slightly (12.52→11.74 m), but speed
  MAE got meaningfully *worse* (3.58→4.54 m/s, +27%) — a single bbox edge is
  noisier frame-to-frame than the box center (detection width jitters at this
  confidence threshold independently of position), and speed is a finite
  difference, which amplifies that noise more than headway's absolute-position
  alignment benefits from the better definitional match. Net effect: a wash,
  not a win, so it's off by default (`pixel_direction_sign=None` uses
  `contact_point` as before) — but it's a real, tested, documented option
  (`run_pipeline.py --pixel-direction-sign`) for anyone who wants to try
  pairing it with more aggressive smoothing, which might recover the speed
  cost while keeping the headway gain.
