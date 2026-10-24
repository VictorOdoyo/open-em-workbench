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


def test_discrete_energy_equals_source_plus_dirichlet_work():
    from emlab.grid import boundary_mask, from_bounds

    grid = from_bounds((0, 1), (0, 1), 19, 19)
    exact, density = m.manufactured_sine(grid, 2)
    potential = m.solve(grid, 2, voltage=exact + 1, charge_density=density)["potential"]
    audit = m.work_balance(grid, potential, 2, boundary_mask(grid), density)
    assert abs(audit["defect"]) < 1e-23


def test_layered_dielectric_has_continuous_displacement_and_discontinuous_electric_field():
    from emlab.grid import from_bounds
    from emlab.operators import harmonic_edges

    grid = from_bounds((0, 1), (0, 0.2), 41, 7)
    epsilon = np.where(grid.mesh()[0] <= 0.5, 2.0, 8.0)
    fixed = np.zeros(grid.shape, bool)
    fixed[:, [0, -1]] = True
    voltage = np.zeros(grid.shape)
    voltage[:, 0] = 1
    potential = m.solve(grid, epsilon, fixed, voltage)["potential"]
    dx, dy = m.displacement(grid, potential, epsilon)
    assert np.ptp(dx) / np.mean(dx) < 1e-12
    assert np.max(abs(dy)) < 1e-23
    ex, _ = m.electric_field(grid, potential)
    eps_face = harmonic_edges(grid, epsilon)[: ex.size].reshape(ex.shape)
    assert np.allclose(ex * eps_face, ex[3, 0] * eps_face[3, 0])


def test_stationary_energy_sensitivity_matches_resolved_finite_difference():
    from emlab.grid import from_bounds

    grid = from_bounds((0, 1), (0, 1), 11, 11)
    voltage = grid.mesh()[0]
    epsilon = np.full(grid.shape, 3.0)
    potential = m.solve(grid, epsilon, voltage=voltage)["potential"]
    derivative = m.energy_sensitivity(grid, potential, epsilon)
    step = 1e-4
    measured = []
    for sign in (-1, 1):
        changed = epsilon.copy()
        changed[5, 6] += sign * step
        field = m.solve(grid, changed, voltage=voltage)["potential"]
        measured.append(m.energy(grid, field, changed))
    assert derivative[5, 6] == pytest.approx((measured[1] - measured[0]) / (2 * step), rel=1e-7)
    assert np.sum(derivative * epsilon) == pytest.approx(m.energy(grid, potential, epsilon))
