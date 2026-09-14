import numpy as np
import pytest

from roadside_headway.calibration.homography import Homography
from roadside_headway.evaluation.auto_calibrate import (
    estimate_pixel_direction_sign,
    fit_homography_by_matching,
    resolve_direction_ambiguity,
)


def test_fit_homography_by_matching_recovers_known_transform_with_noise_and_distractors():
    rng = np.random.default_rng(42)

    # True mapping: world = 0.2 * pixel + (5, -3) -- easy to reason about, and
    # deliberately not aligned with pixel axes growing in the "obvious" direction.
    true_homography = Homography.fit(
        [(0, 0), (100, 0), (0, 100), (100, 100)],
        [(5, -3), (25, -3), (5, 17), (25, 17)],
    )

    frames_pixel_points: dict[int, list[tuple[float, float]]] = {}
    frames_world_points: dict[int, list[tuple[float, float]]] = {}

    for frame_idx in range(20):
        n_real = rng.integers(3, 6)
        pixel_pts = rng.uniform(0, 100, size=(n_real, 2))
        world_pts = [true_homography.pixel_to_world(tuple(p)) for p in pixel_pts]
        noisy_world_pts = [(w[0] + rng.normal(0, 0.05), w[1] + rng.normal(0, 0.05)) for w in world_pts]

        # distractor ground-truth points with no matching pixel detection this frame
        n_distractors = rng.integers(0, 3)
        distractors = [tuple(rng.uniform(-10, 40, size=2)) for _ in range(n_distractors)]

        # shuffle world points so pairing order carries no information
        combined = noisy_world_pts + distractors
        rng.shuffle(combined)

        frames_pixel_points[frame_idx] = [tuple(p) for p in pixel_pts]
        frames_world_points[frame_idx] = combined

    fitted_homography, stats = fit_homography_by_matching(
        frames_pixel_points, frames_world_points, initial_threshold_m=15.0, final_threshold_m=1.0
    )

    assert stats["mean_error_m"] < 0.5
    assert stats["n_matched"] > 30

    for test_pixel in [(10, 10), (50, 50), (90, 20)]:
        expected = true_homography.pixel_to_world(test_pixel)
        got = fitted_homography.pixel_to_world(test_pixel)
        assert abs(got[0] - expected[0]) < 0.5
        assert abs(got[1] - expected[1]) < 0.5


def test_fit_homography_by_matching_requires_minimum_points():
    with pytest.raises(ValueError):
        fit_homography_by_matching({0: [(0, 0), (1, 1)]}, {0: [(0, 0), (1, 1)]})


def test_resolve_direction_ambiguity_flips_when_tracks_move_backward():
    # A homography whose world_x decreases as pixel_x increases -- i.e. tracks
    # moving toward larger pixel_x (as they would on screen, left to right)
    # end up moving toward *smaller* world_x, the "wrong" direction.
    backward_homography = Homography.fit(
        [(0, 0), (100, 0), (0, 100), (100, 100)],
        [(50, 0), (0, 0), (50, 10), (0, 10)],
    )
    track_pixel_sequences = {
        1: [(10, 5), (30, 5), (50, 5), (70, 5), (90, 5)],
        2: [(20, 8), (40, 8), (60, 8), (80, 8)],
    }

    fixed = resolve_direction_ambiguity(backward_homography, track_pixel_sequences)

    xs = [fixed.pixel_to_world(p)[0] for p in track_pixel_sequences[1]]
    assert xs[-1] > xs[0]  # now moves toward increasing world_x

    # Direction must flip WITHOUT shifting the absolute coordinate range —
    # a plain negation would reverse direction correctly but also move
    # everything into negative territory, breaking comparison against a
    # ground truth that uses absolute (positive) positions.
    original_range = sorted(wx for wx, _ in backward_homography.dest_points)
    fixed_range = sorted(wx for wx, _ in fixed.dest_points)
    assert fixed_range == pytest.approx(original_range)


def test_resolve_direction_ambiguity_leaves_correct_homography_unchanged():
    forward_homography = Homography.fit(
        [(0, 0), (100, 0), (0, 100), (100, 100)],
        [(0, 0), (50, 0), (0, 10), (50, 10)],
    )
    track_pixel_sequences = {1: [(10, 5), (30, 5), (50, 5), (70, 5), (90, 5)]}

    result = resolve_direction_ambiguity(forward_homography, track_pixel_sequences)

    assert result.pixel_to_world((10, 5)) == forward_homography.pixel_to_world((10, 5))


def test_resolve_direction_ambiguity_ignores_short_tracks():
    forward_homography = Homography.fit(
        [(0, 0), (100, 0), (0, 100), (100, 100)],
        [(0, 0), (50, 0), (0, 10), (50, 10)],
    )
    # only 2 points -- below min_track_length, should not influence the decision
    track_pixel_sequences = {1: [(90, 5), (10, 5)]}

    result = resolve_direction_ambiguity(forward_homography, track_pixel_sequences)

    assert result.pixel_to_world((10, 5)) == forward_homography.pixel_to_world((10, 5))


def test_estimate_pixel_direction_sign_majority_positive():
    sequences = {
        1: [(10, 5), (30, 5), (50, 5), (70, 5), (90, 5)],  # increasing
        2: [(20, 8), (40, 8), (60, 8), (80, 8), (100, 8)],  # increasing
        3: [(90, 3), (10, 3), (5, 3), (2, 3), (1, 3)],  # decreasing (minority)
    }
    assert estimate_pixel_direction_sign(sequences) == 1


def test_estimate_pixel_direction_sign_majority_negative():
    sequences = {
        1: [(90, 5), (70, 5), (50, 5), (30, 5), (10, 5)],
        2: [(100, 8), (80, 8), (60, 8), (40, 8), (20, 8)],
    }
    assert estimate_pixel_direction_sign(sequences) == -1


def test_estimate_pixel_direction_sign_ignores_short_tracks():
    sequences = {1: [(90, 5), (10, 5)]}  # only 2 points, below default min_track_length
    assert estimate_pixel_direction_sign(sequences) == 1  # falls back to default


def test_estimate_pixel_direction_sign_empty_input():
    assert estimate_pixel_direction_sign({}) == 1
