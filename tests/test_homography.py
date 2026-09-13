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


def test_robust_fit_rejects_outlier_correspondence():
    rng = np.random.default_rng(0)
    # A well-behaved grid of correspondences: world = 0.1 * pixel + small noise.
    pixel_points = [(x, y) for x in range(0, 101, 10) for y in range(0, 101, 10)]
    world_points = [(0.1 * x + rng.normal(0, 0.01), 0.1 * y + rng.normal(0, 0.01)) for x, y in pixel_points]
    # One wrong correspondence: this pixel should map near (5.0, 5.0), not (500.0, 500.0).
    pixel_points.append((50, 50))
    world_points.append((500.0, 500.0))

    plain = Homography.fit(pixel_points, world_points, robust=False)
    robust = Homography.fit(pixel_points, world_points, robust=True)

    # The outlier should meaningfully corrupt the plain fit's prediction near
    # its own location, while the robust fit stays close to the true mapping.
    plain_x, plain_y = plain.pixel_to_world((50, 50))
    robust_x, robust_y = robust.pixel_to_world((50, 50))
    assert abs(robust_x - 5.0) < 0.5
    assert abs(robust_y - 5.0) < 0.5
    assert abs(plain_x - 5.0) > abs(robust_x - 5.0)

    # At least the outlier pair should have been dropped from the robust fit.
    assert len(robust.source_points) < len(pixel_points)


def test_save_and_load_round_trip(tmp_path):
    pixel_points = [(0, 0), (100, 0), (0, 100), (100, 100)]
    world_points = [(0, 0), (10, 0), (0, 10), (10, 10)]
    h = Homography.fit(pixel_points, world_points)

    path = tmp_path / "homography.json"
    h.save(path)
    loaded = Homography.load(path)

    np.testing.assert_allclose(loaded.matrix, h.matrix)
    assert loaded.pixel_to_world((50, 50)) == pytest.approx(h.pixel_to_world((50, 50)))
