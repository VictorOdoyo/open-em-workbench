import numpy as np

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
