import numpy as np

from roadside_headway.calibration.homography import Homography
from roadside_headway.evaluation.auto_calibrate import fit_homography_by_matching


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
    import pytest

    with pytest.raises(ValueError):
        fit_homography_by_matching({0: [(0, 0), (1, 1)]}, {0: [(0, 0), (1, 1)]})
