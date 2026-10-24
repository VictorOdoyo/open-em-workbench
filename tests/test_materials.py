import numpy as np
import pytest

from emlab import materials as m


def test_passive_conductivity_has_negative_imaginary_permittivity():
    from emlab.core import EPS0

    eps = m.permittivity(1e9, 4, 0.1)
    assert eps.real == 4 and eps.imag < 0
    assert -eps.imag * (2 * np.pi * 1e9) * EPS0 == pytest.approx(0.1)
    with pytest.raises(ValueError):
        m.permittivity(0)


def test_propagation_decays_and_satisfies_maxwell_impedance_relation():
    from emlab.core import C0, ETA0

    gamma, eta = m.propagation(1e9, 4)
    assert gamma == pytest.approx(1j * 4 * np.pi * 1e9 / C0)
    assert eta == pytest.approx(ETA0 / 2)
    lossy, impedance = m.propagation(1e9, m.permittivity(1e9, 4, 1))
    assert lossy.real > 0 and impedance.real > 0
    evanescent, _ = m.propagation(1e9, -4)
    assert evanescent.real > 0 and abs(evanescent.imag) < 1e-12
