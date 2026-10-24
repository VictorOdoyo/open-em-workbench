import numpy as np
import pytest

from emlab import layers as m


def test_interface_enforces_tangential_electric_and_magnetic_continuity():
    scattering = m.interface(1 / 50, 1 / 100)
    r, t = scattering[0, 0], scattering[1, 0]
    assert 1 + r == pytest.approx(t)
    assert (1 - r) / 50 == pytest.approx(t / 100)


def test_layer_propagation_handles_optically_thick_absorbers_without_overflow():
    result = m.propagation(3 - 1e6j, 1)
    assert np.all(np.isfinite(result)) and np.all(result == 0)
    vacuum = m.propagation(2, 0.3)
    assert vacuum[1, 0] == pytest.approx(np.exp(-0.6j))
