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
