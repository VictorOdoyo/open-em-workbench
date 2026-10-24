import numpy as np
import pytest

from emlab import design as m


def test_bounded_material_projection_satisfies_budget_bounds_and_stationarity():
    values = np.array([-1.0, 0.5, 1.5, 5.0])
    weights = np.array([1.0, 2, 1, 3])
    projected = m.bounded_mean(values, 1, 4, 2.5, weights)
    assert projected.min() >= 1 and projected.max() <= 4
    assert np.sum(projected * weights) / weights.sum() == pytest.approx(2.5, abs=1e-13)
    interior = (projected > 1 + 1e-12) & (projected < 4 - 1e-12)
    assert np.ptp((projected - values)[interior]) < 1e-12
