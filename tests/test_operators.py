import numpy as np

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
