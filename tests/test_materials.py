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
