import numpy as np
import pytest

from emlab import modes as m


def test_rectangular_cutoff_matches_standard_te10_and_excludes_spurious_tem():
    from emlab.core import C0

    assert m.rectangular_cutoff(0.02, 0.01, 1, 0) == pytest.approx(C0 / 0.04)
    with pytest.raises(ValueError):
        m.rectangular_cutoff(0.02, 0.01, 0, 0)
    with pytest.raises(ValueError):
        m.rectangular_cutoff(0.02, 0.01, 1, 0, "tm")


def test_generalized_cavity_modes_match_independent_rectangular_cutoffs():
    from emlab.grid import from_bounds

    grid = from_bounds((0, 0.02), (0, 0.01), 33, 25)
    for polarization, indices in (("tm", (1, 1)), ("te", (1, 0))):
        result = m.cavity_modes(grid, epsilon_r=4, polarization=polarization, count=3)
        expected = m.rectangular_cutoff(0.02, 0.01, *indices, polarization, epsilon_r=4)
        assert result["cutoff_hz"][0] == pytest.approx(expected, rel=0.002)
        assert np.max(result["residuals"]) < 1e-8
        assert np.all(np.diff(result["cutoff_hz"]) >= 0)


def test_tm_mode_gram_and_mesh_order_are_independent_of_eigenvector_scaling():
    from emlab.core import convergence_order
    from emlab.grid import from_bounds

    exact = m.rectangular_cutoff(0.02, 0.01, 1, 1, "tm", 3)
    errors, spacing = [], []
    for count in (13, 25, 49):
        grid = from_bounds((0, 0.02), (0, 0.01), count, count)
        result = m.cavity_modes(grid, 3, count=2)
        assert np.allclose(m.mode_gram(grid, result["fields"], 3), np.eye(2), atol=1e-10)
        errors.append(abs(result["cutoff_hz"][0] - exact))
        spacing.append(grid.dx)
    assert 1.95 < convergence_order(spacing, errors)[0] < 2.05


def test_below_cutoff_fields_decay_and_above_cutoff_fields_accumulate_phase():
    gamma = m.guide_propagation(np.array([5e9, 15e9]), 10e9)
    assert gamma[0].real > 0 and gamma[0].imag == 0
    assert gamma[1].imag > 0 and gamma[1].real == 0
    assert abs(np.exp(-gamma[0] * 0.1)) < 0.001


def test_te_tm_impedances_have_dual_cutoff_behavior():
    from emlab.core import ETA0

    frequency, cutoff = 15e9, 10e9
    te = m.guide_impedance(frequency, cutoff, "te")
    tm = m.guide_impedance(frequency, cutoff, "tm")
    assert te.real > ETA0 and 0 < tm.real < ETA0
    assert te * tm == pytest.approx(ETA0**2)
    assert m.guide_impedance(5e9, cutoff, "te").imag > 0
    assert m.guide_impedance(5e9, cutoff, "tm").imag < 0


def test_waveguide_velocity_product_does_not_imply_superluminal_information():
    from emlab.core import C0

    phase, group = m.guide_velocities(15e9, 10e9)
    assert phase > C0 and group < C0
    assert phase * group == pytest.approx(C0**2)


def test_modal_projection_recovers_complex_amplitudes_without_assuming_unit_norm():
    from emlab.grid import from_bounds

    grid = from_bounds((0, 1), (0, 1), 17, 17)
    x, y = grid.mesh()
    first = np.sin(np.pi * x) * np.sin(np.pi * y)
    second = np.sin(2 * np.pi * x) * np.sin(np.pi * y) + 0.2 * first
    basis = np.stack([first, second])
    expected = np.array([2 + 1j, -0.4 + 0.2j])
    field = np.einsum("i,iyx->yx", expected, basis)
    measured, recovered = m.project(grid, field, basis)
    assert np.allclose(measured, expected)
    assert np.allclose(recovered, field)


def test_degenerate_subspace_metric_is_invariant_to_unitary_mode_mixing():
    from emlab.grid import from_bounds

    grid = from_bounds((0, 1), (0, 1), 17, 17)
    x, y = grid.mesh()
    first = np.stack([np.sin(np.pi * x) * np.sin(2 * np.pi * y), np.sin(2 * np.pi * x) * np.sin(np.pi * y)])
    rotation = np.array([[1, 1j], [1j, 1]]) / np.sqrt(2)
    second = np.einsum("ij,jyx->iyx", rotation, first)
    assert np.allclose(m.subspace_overlap(grid, first, second), 1)


def test_first_order_cavity_perturbation_matches_a_resolved_dielectric_change():
    from emlab.grid import from_bounds

    grid = from_bounds((0, 0.02), (0, 0.01), 29, 23)
    initial = m.cavity_modes(grid, count=1)
    delta = np.zeros(grid.shape)
    delta[8:15, 10:19] = 1e-4
    predicted = m.dielectric_shift(grid, initial["fields"][0], 1, delta)
    changed = m.cavity_modes(grid, 1 + delta, count=1)
    measured = changed["cutoff_hz"][0] / initial["cutoff_hz"][0] - 1
    assert measured == pytest.approx(predicted, rel=1e-4)
