import numpy as np
import pytest

from emlab import core as m


def test_vacuum_constants_and_frequency_wavelength_identity():
    assert m.EPS0 * m.MU0 * m.C0**2 == pytest.approx(1)
    assert m.ETA0 == pytest.approx(376.7303134)
    assert m.wavelength(1e9) == pytest.approx(0.299792458)
    assert np.allclose(m.frequency(m.wavelength([1e6, 1e9])), [1e6, 1e9])
    with pytest.raises(ValueError):
        m.wavelength(0)
