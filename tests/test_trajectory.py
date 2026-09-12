import numpy as np
import pytest

from roadside_headway.headway.trajectory import compute_speed, extrapolate_constant_velocity, smooth_positions


def test_extrapolate_constant_velocity():
    history = [(0.0, 0.0), (1.0, 2.0)]
    assert extrapolate_constant_velocity(history) == (2.0, 4.0)


def test_smooth_positions_preserves_shape_and_ends():
    positions = np.array([[float(i), float(2 * i)] for i in range(10)])
    smoothed = smooth_positions(positions, window=3)
    assert smoothed.shape == positions.shape
    np.testing.assert_allclose(smoothed[0], positions[0])
    np.testing.assert_allclose(smoothed[-1], positions[-1])


def test_smooth_positions_short_input_returns_copy():
    positions = np.array([[0.0, 0.0], [1.0, 1.0]])
    smoothed = smooth_positions(positions, window=5)
    np.testing.assert_allclose(smoothed, positions)


def test_compute_speed_constant_velocity():
    times = np.linspace(0, 9, 10)
    longitudinal = 3.0 * times  # 3 m/s constant speed
    speed = compute_speed(times, longitudinal)
    np.testing.assert_allclose(speed, 3.0, atol=1e-9)


def test_compute_speed_single_sample_is_nan():
    speed = compute_speed(np.array([0.0]), np.array([1.0]))
    assert np.isnan(speed).all()
