import numpy as np
import pytest

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


def test_gauss_charge_reaction_balances_source_charge_and_ground():
    from emlab.core import EPS0
    from emlab.grid import boundary_mask, dual_weights, from_bounds

    grid = from_bounds((0, 1), (0, 1), 21, 21)
    density = np.zeros(grid.shape)
    density[10, 10] = EPS0 / grid.dx / grid.dy
    result = m.solve(grid, charge_density=density)
    reaction = m.charge_reaction(grid, result["potential"], charge_density=density)
    fixed = boundary_mask(grid)
    assert np.max(abs(reaction[~fixed])) < 1e-24
    assert reaction[fixed].sum() == pytest.approx(-np.sum(density * dual_weights(grid)), rel=1e-12)


def test_uniform_field_energy_matches_continuum_volume_integral():
    from emlab.core import EPS0
    from emlab.grid import from_bounds

    grid = from_bounds((0, 2), (0, 1), 17, 13)
    x, _ = grid.mesh()
    assert m.energy(grid, 3 * x, 4) == pytest.approx(0.5 * EPS0 * 4 * 9 * 2, rel=1e-13)


def test_conductor_charge_and_field_energy_agree_for_parallel_plates():
    from emlab.core import EPS0
    from emlab.grid import from_bounds

    grid = from_bounds((0, 2), (0, 1), 21, 11)
    labels = np.full(grid.shape, -1, int)
    labels[0] = 0
    labels[-1] = 1
    result = m.solve(grid, 3, labels >= 0, (labels == 1) * 2)
    charges = m.conductor_charges(grid, labels, result["potential"], 3)
    expected = EPS0 * 3 * 2 / 1
    assert charges[1] == pytest.approx(expected * 2, rel=1e-12)
    assert charges[0] + charges[1] == pytest.approx(0, abs=1e-24)
    assert m.energy(grid, result["potential"], 3) == pytest.approx(0.5 * 2 * charges[1])


def test_manufactured_solution_converges_quadratically_under_mesh_refinement():
    from emlab.core import convergence_order, relative_error
    from emlab.grid import from_bounds

    spacings, errors = [], []
    for nodes in (13, 25, 49):
        grid = from_bounds((0, 1), (0, 1), nodes, nodes)
        exact, density = m.manufactured_sine(grid, 3)
        potential = m.solve(grid, 3, voltage=exact, charge_density=density)["potential"]
        spacings.append(grid.dx)
        errors.append(relative_error(exact, potential))
    assert 1.95 < convergence_order(spacings, errors)[0] < 2.05
