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
