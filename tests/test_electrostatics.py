import numpy as np

from emlab import electrostatics as m


def test_poisson_recovers_affine_harmonic_potential():
    from emlab.grid import from_bounds

    grid = from_bounds((0, 2), (0, 1), 21, 15)
    x, y = grid.mesh()
    expected = 3 * x - 2 * y + 1
    result = m.solve(grid, epsilon_r=4, voltage=expected)
    assert np.max(abs(result["potential"] - expected)) < 1e-12
    assert result["residual"] < 1e-12


def test_electric_field_sign_and_axis_order_follow_minus_gradient():
    from emlab.grid import Grid2D

    grid = Grid2D(9, 7, 0.2, 0.3)
    x, y = grid.mesh()
    for function in (m.electric_field, m.nodal_field):
        ex, ey = function(grid, 4 * x - 3 * y)
        assert np.allclose(ex, -4) and np.allclose(ey, 3)
