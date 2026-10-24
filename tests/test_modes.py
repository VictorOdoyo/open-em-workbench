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
