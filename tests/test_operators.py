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


def test_stiffness_is_symmetric_positive_semidefinite_and_conservative():
    from emlab.grid import Grid2D

    grid = Grid2D(8, 6, 0.1, 0.2)
    operator = m.stiffness(grid, 3)
    assert np.allclose(operator.toarray(), operator.toarray().T)
    assert np.max(abs(operator @ np.ones(48))) < 1e-14
    rng = np.random.default_rng(4)
    field = rng.normal(size=48)
    assert field @ operator @ field > 0


def test_dirichlet_elimination_uses_boundary_load_without_row_replacement():
    from scipy.sparse import csr_matrix

    matrix = csr_matrix([[2.0, -1.0, 0], [-1, 2, -1], [0, -1, 2]])
    reduced, load, free, base = m.eliminate_dirichlet(matrix, np.zeros(3), [True, False, True], [1, 0, 3])
    assert reduced.toarray()[0, 0] == 2
    assert load[0] == 4 and np.array_equal(free, [1])
    assert np.array_equal(base, [1, 0, 3])
