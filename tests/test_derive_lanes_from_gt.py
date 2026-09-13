import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from derive_lanes_from_gt import derive_boundaries  # noqa: E402


def test_derive_boundaries_from_synthetic_lanes(tmp_path):
    # Three lanes at local_x = 10, 20, 30 ft -> medians 3.048, 6.096, 9.144 m
    rows = []
    for lane_id, local_x in [(1, 10.0), (2, 20.0), (3, 30.0)]:
        for _ in range(5):
            rows.append({"lane_id": lane_id, "local_x": local_x, "global_x": 6451900.0, "global_y": 1872650.0})
    csv_path = tmp_path / "gt.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    boundaries = derive_boundaries(str(csv_path), camera=4)

    ft_to_m = 0.3048
    expected_medians = [10.0 * ft_to_m, 20.0 * ft_to_m, 30.0 * ft_to_m]
    expected = [
        expected_medians[0] - (expected_medians[1] - expected_medians[0]) / 2,
        (expected_medians[0] + expected_medians[1]) / 2,
        (expected_medians[1] + expected_medians[2]) / 2,
        expected_medians[2] + (expected_medians[2] - expected_medians[1]) / 2,
    ]
    for got, want in zip(boundaries, expected):
        assert abs(got - want) < 1e-9


def test_derive_boundaries_requires_at_least_two_lanes(tmp_path):
    import pytest

    rows = [{"lane_id": 1, "local_x": 10.0, "global_x": 6451900.0, "global_y": 1872650.0}]
    csv_path = tmp_path / "gt.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    with pytest.raises(ValueError):
        derive_boundaries(str(csv_path), camera=4)
