import numpy as np
import pytest

from emlab import transmission as m


def test_lossless_line_velocity_and_impedance_follow_independent_telegrapher_relations():
    frequency, inductance, capacitance = 1e9, 250e-9, 100e-12
    gamma, impedance = m.rlgc(frequency, 0, inductance, 0, capacitance)
    assert impedance == pytest.approx(50)
    assert gamma == pytest.approx(2j * np.pi * frequency * np.sqrt(inductance * capacitance))
    lossy, _ = m.rlgc(frequency, 10, inductance, 1e-3, capacitance)
    assert lossy.real > 0 and lossy.imag > 0


def test_line_chain_determinant_and_zero_length_are_exact_physical_constraints():
    gamma = 0.1 + 2j
    matrix = m.line_matrix(gamma, 75, 0.3)
    assert np.linalg.det(matrix) == pytest.approx(1)
    assert np.allclose(m.line_matrix(gamma, 75, 0), np.eye(2))


def test_stable_line_scattering_matches_chain_reference_and_survives_thick_loss():
    from emlab.networks import abcd_to_s

    gamma, impedance = m.rlgc(np.linspace(1e8, 2e8, 5), 4, 300e-9, 0.001, 100e-12)
    expected = abcd_to_s(m.line_matrix(gamma, impedance, 0.7))
    assert np.allclose(m.line_scattering(gamma, impedance, 0.7), expected)
    thick = m.line_scattering(1e5 + 1j, 60 + 3j, 1)
    assert np.all(np.isfinite(thick))
    assert thick[1, 0] == 0


def test_quarter_wave_line_inverts_resistive_load_and_swaps_open_short():
    gamma = 1j * np.pi / 2
    assert m.input_impedance(gamma, 50, 1, 100) == pytest.approx(25)
    assert abs(m.input_impedance(gamma, 50, 1, 0)) > 1e15
    assert abs(m.input_impedance(gamma, 50, 1, np.inf)) < 1e-12
