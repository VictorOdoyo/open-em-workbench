import numpy as np

from emlab import networks as m


def test_series_and_shunt_chain_equations_obey_voltage_current_relations():
    voltage, current = 2 + 1j, 0.3 - 0.1j
    assert np.allclose(m.series(5) @ [voltage, current], [voltage + 5 * current, current])
    assert np.allclose(m.shunt(0.2) @ [voltage, current], [voltage, current + 0.2 * voltage])
