import numpy as np
import pytest

from emlab import operators as m


def test_incidence_has_constant_nullspace_and_oriented_edge_differences():
    from emlab.grid import Grid2D

    grid = Grid2D(5, 4, 0.2, 0.3)
    operator = m.incidence(grid)
    x, y = grid.mesh()
    assert np.all(operator @ np.ones(20) == 0)
    values = operator @ (2 * x + 3 * y).ravel()
    assert np.allclose(values[:16], 2 * grid.dx)
    assert np.allclose(values[16:], 3 * grid.dy)


def test_edge_geometry_has_half_faces_only_on_domain_edges():
    from emlab.grid import Grid2D

    lengths, areas = m.edge_geometry(Grid2D(5, 4, 0.2, 0.3))
    assert len(lengths) == len(areas) == 31
    assert areas[0] == pytest.approx(0.15)
    assert areas[4] == pytest.approx(0.3)
    assert lengths[-1] == pytest.approx(0.3)


def test_interface_average_matches_two_series_half_cells():
    from emlab.grid import Grid2D

    grid = Grid2D(3, 3, 1, 1)
    material = np.tile([2, 8, 8], (3, 1))
    edges = m.harmonic_edges(grid, material)
    assert edges[0] == pytest.approx(3.2)
    assert edges[1] == pytest.approx(8)
    with pytest.raises(ValueError):
        m.harmonic_edges(grid, 0)
