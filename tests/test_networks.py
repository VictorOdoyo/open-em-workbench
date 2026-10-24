import numpy as np
import pytest

from emlab import networks as m


def test_series_and_shunt_chain_equations_obey_voltage_current_relations():
    voltage, current = 2 + 1j, 0.3 - 0.1j
    assert np.allclose(m.series(5) @ [voltage, current], [voltage + 5 * current, current])
    assert np.allclose(m.shunt(0.2) @ [voltage, current], [voltage, current + 0.2 * voltage])


def test_cascade_keeps_noncommuting_physical_order_and_frequency_batches():
    result = m.cascade(m.series(np.array([1, 2])), m.shunt(0.5))
    assert result.shape == (2, 2, 2)
    assert np.allclose(result[0], [[1.5, 1], [0.5, 1]])
    assert not np.allclose(m.cascade(m.series(1), m.shunt(0.5)), m.cascade(m.shunt(0.5), m.series(1)))


def test_impedance_step_is_unitary_with_power_wave_normalization():
    scattering = m.abcd_to_s(np.eye(2), (50, 100))
    assert scattering[0, 0] == pytest.approx(1 / 3)
    assert scattering[1, 0] == pytest.approx(2 * np.sqrt(2) / 3)
    assert np.allclose(scattering.conj().T @ scattering, np.eye(2))


def test_chain_conversion_matches_known_series_element_with_unequal_ports():
    matrix = m.series(12 + 3j)
    scattering = m.abcd_to_s(matrix, (35, 80))
    assert np.allclose(m.s_to_abcd(scattering, (35, 80)), matrix)
    with pytest.raises(ValueError, match="too small"):
        m.s_to_abcd(np.zeros((2, 2)))


def test_redheffer_product_matches_chain_cascade_and_handles_extreme_attenuation():
    first, second = m.series(20), m.shunt(0.01j)
    expected = m.abcd_to_s(m.cascade(first, second))
    assert np.allclose(m.star(m.abcd_to_s(first), m.abcd_to_s(second)), expected)
    attenuator = np.array([[0, 1e-100], [1e-100, 0]], complex)
    result = m.star(attenuator, attenuator)
    assert np.all(np.isfinite(result))
    assert result[1, 0] == pytest.approx(1e-200, abs=0, rel=1e-12)


def test_renormalization_matches_unequal_reference_chain_formula_even_for_ideal_through():
    for matrix in (np.eye(2), m.series(25 + 4j)):
        original = m.abcd_to_s(matrix, (50, 50))
        result = m.renormalize(original, 50, (30, 100))
        assert np.allclose(result, m.abcd_to_s(matrix, (30, 100)))


def test_terminated_series_network_matches_direct_input_impedance():
    series = m.abcd_to_s(m.series(25))
    load = (100 - 50) / (100 + 50)
    assert m.terminated_input(series, load) == pytest.approx((125 - 50) / (125 + 50))


def test_open_short_matching_and_standing_wave_ratio():
    assert np.allclose(m.reflection([0, 50, np.inf]), [-1, 0, 1])
    assert m.standing_wave_ratio(0.5) == pytest.approx(3)
    assert np.isinf(m.standing_wave_ratio(1))
    assert abs(m.reflection(1j * 50)) == pytest.approx(1)


def test_scattering_audit_detects_active_gain_and_nonreciprocity():
    resistor = m.scattering_audit(m.abcd_to_s(m.series(50)))
    assert resistor["passivity_margin"] >= -1e-14
    assert resistor["reciprocity_error"] < 1e-14
    active = np.array([[0, 0], [2, 0]], complex)
    audit = m.scattering_audit(active)
    assert audit["passivity_margin"] == pytest.approx(-3)
    assert audit["reciprocity_error"] > 0
