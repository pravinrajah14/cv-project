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
#    same data the homography was fit to:
python scripts/run_pipeline.py data/sb-camera4-0750am-0805am.avi \
    --homography results/homography.json --lanes results/lanes.json \
    --start-frame 900 --max-frames 900 \
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

From the reference run: US-101 camera 4, `0750am-0805am`. Homography fit
on frames 0–899 (794 matched points, 3.02 m mean fitting residual), evaluated
on the held-out frames 900–1799 (a different 90-second span the fit never
saw). Full numbers in `results/metrics.json`.

| Metric | MAE | MAPE | Bias | n |
|---|---|---|---|---|
| Speed | 5.66 m/s (12.7 mph) | 42.2% | −2.63 m/s | 346 |
| Headway | 15.04 m | 62.9% | −10.50 m | 134 |

Take these as a first real data point, not a general accuracy claim — see
[Limitations](#limitations) for why: NGSIM's own ground truth carries
reconstruction noise, the two headway definitions differ systematically
(the negative bias here is consistent with that, not just error), and this
covers one camera, one 90-second span, one calibration run. Two bugs were
caught and fixed only by actually running this evaluation against real
ground truth — a ground-truth axis mix-up that silently zeroed out every
match, and a direction-of-travel sign ambiguity inherent to fitting a
homography from position snapshots alone (both in the git history) — which is
itself a reason to treat any calibration/validation code, including this
run's, with some skepticism until it's been exercised end to end.

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
- **Single-camera, single-segment validation.** Results describe this one
  camera/segment, not the pipeline's general accuracy across conditions.
- **Auto-calibration's direction check covers longitudinal only.**
  `calibrate_from_gt.py` resolves the direction-of-travel mirror ambiguity
  (see Results) using tracked motion, but doesn't run the equivalent check on
  the lateral axis — a lateral mirror is left undetected if the ICP fit
  happens to converge to one. It didn't in this run (per-lane metrics came
  out sane), but that's an empirical observation for this footage, not a
  guarantee for other camera views.
