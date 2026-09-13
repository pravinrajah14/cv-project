import numpy as np
import pytest

from roadside_headway.calibration.homography import Homography


def _sample_homography() -> Homography:
    # world = 0.1 * pixel, dest_points span x in [0,10], y in [0,10]
    return Homography.fit(
        [(0, 0), (100, 0), (0, 100), (100, 100)],
        [(0, 0), (10, 0), (0, 10), (10, 10)],
    )


def test_calibrated_bounds_expands_by_margin_fraction():
    homography = _sample_homography()
    (x_lo, x_hi), (y_lo, y_hi) = homography.calibrated_bounds(margin_frac=1.0)
    # span is 10 on each axis, margin_frac=1.0 doubles it on each side
    assert x_lo == pytest.approx(-10.0)
    assert x_hi == pytest.approx(20.0)
    assert y_lo == pytest.approx(-10.0)
    assert y_hi == pytest.approx(20.0)


def test_is_within_calibrated_region_accepts_nearby_points():
    homography = _sample_homography()
    assert homography.is_within_calibrated_region((5.0, 5.0), margin_frac=1.0)
    assert homography.is_within_calibrated_region((-9.0, -9.0), margin_frac=1.0)


def test_is_within_calibrated_region_rejects_far_extrapolation():
    homography = _sample_homography()
    assert not homography.is_within_calibrated_region((10000.0, 5.0), margin_frac=1.0)
    assert not homography.is_within_calibrated_region((5.0, -500.0), margin_frac=1.0)


def test_calibrated_bounds_handles_no_points():
    homography = Homography(matrix=np.eye(3))
    (x_lo, x_hi), (y_lo, y_hi) = homography.calibrated_bounds()
    assert x_lo == float("-inf")
    assert x_hi == float("inf")
