import numpy as np
import pytest

from emlab import grid as m


def test_cartesian_grid_uses_physical_coordinates_and_consistent_shape():
    grid = m.Grid2D(7, 5, 0.2, 0.3, -0.6, 1.0)
    x, y = grid.mesh()
    assert x.shape == y.shape == grid.shape == (5, 7)
    assert x[2, 3] == pytest.approx(0)
    assert y[2, 3] == pytest.approx(1.6)
    with pytest.raises(ValueError):
        m.Grid2D(1, 8, 1, 1)


def test_bounds_constructor_preserves_both_physical_endpoints():
    grid = m.from_bounds((-1, 2), (0, 4), 13, 17)
    assert grid.x[-1] == pytest.approx(2)
    assert grid.y[-1] == pytest.approx(4)


def test_dual_cell_weights_integrate_domain_and_linear_fields_exactly():
    grid = m.from_bounds((0, 2), (-1, 2), 9, 7)
    x, y = grid.mesh()
    weights = m.dual_weights(grid)
    assert weights.sum() == pytest.approx(6)
    assert np.sum(weights * x) == pytest.approx(6)
    assert np.sum(weights * y) == pytest.approx(3)


def test_boundary_mask_counts_corners_once():
    grid = m.Grid2D(8, 6, 1, 1)
    mask = m.boundary_mask(grid)
    assert mask.sum() == 2 * 8 + 2 * 6 - 4
    assert not mask[2, 3]


def test_probe_mapping_rejects_outside_locations_instead_of_clamping():
    grid = m.Grid2D(5, 7, 0.2, 0.1)
    assert m.nearest_index(grid, 0.39, 0.29) == (3, 2)
    with pytest.raises(ValueError):
        m.nearest_index(grid, -0.1, 0)
