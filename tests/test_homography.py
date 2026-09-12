import numpy as np
import pytest

from roadside_headway.calibration.homography import Homography


def test_fit_recovers_simple_affine_mapping(tmp_path):
    # world = 0.1 * pixel (a pure scale, easy to reason about)
    pixel_points = [(0, 0), (100, 0), (0, 100), (100, 100)]
    world_points = [(0, 0), (10, 0), (0, 10), (10, 10)]

    h = Homography.fit(pixel_points, world_points)

    x, y = h.pixel_to_world((50, 50))
    assert x == pytest.approx(5.0, abs=1e-6)
    assert y == pytest.approx(5.0, abs=1e-6)
    assert h.reprojection_error_m() == pytest.approx(0.0, abs=1e-6)


def test_fit_requires_at_least_four_points():
    with pytest.raises(ValueError):
        Homography.fit([(0, 0), (1, 0), (0, 1)], [(0, 0), (1, 0), (0, 1)])


def test_save_and_load_round_trip(tmp_path):
    pixel_points = [(0, 0), (100, 0), (0, 100), (100, 100)]
    world_points = [(0, 0), (10, 0), (0, 10), (10, 10)]
    h = Homography.fit(pixel_points, world_points)

    path = tmp_path / "homography.json"
    h.save(path)
    loaded = Homography.load(path)

    np.testing.assert_allclose(loaded.matrix, h.matrix)
    assert loaded.pixel_to_world((50, 50)) == pytest.approx(h.pixel_to_world((50, 50)))
