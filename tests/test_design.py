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


def test_material_design_improves_capacitance_without_adding_dielectric_volume():
    from emlab.geometry import conductor_labels
    from emlab.grid import Grid2D, dual_weights

    grid = Grid2D(25, 21, 0.001, 0.001)
    electrode = np.zeros(grid.shape, bool)
    electrode[7:14, 11:14] = True
    labels = conductor_labels(grid, [electrode])
    mask = labels < 0
    result = m.optimize_dielectric(grid, labels >= 0, (labels == 1).astype(float), mask, iterations=6)
    history = np.array([r["energy_j_m"] for r in result["history"]])
    assert history[-1] > history[0] * 1.03
    assert np.all(np.diff(history) >= 0)
    epsilon = result["epsilon_r"]
    weights = dual_weights(grid)[mask]
    assert np.sum(epsilon[mask] * weights) / weights.sum() == pytest.approx(2.5, abs=1e-12)
    assert np.min(epsilon) >= 1 and np.max(epsilon) <= 4


@pytest.mark.parametrize("bounds", [(0, 4), (-1, 4), (4, 1), (1, float("inf"))])
def test_design_rejects_nonphysical_material_bounds(bounds):
    from emlab.grid import Grid2D, boundary_mask

    grid = Grid2D(7, 7, 0.001, 0.001)
    fixed = boundary_mask(grid)
    with pytest.raises(ValueError, match="bounds"):
        m.optimize_dielectric(grid, fixed, 0, ~fixed, bounds=bounds)
