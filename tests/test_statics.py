import numpy as np
import pytest

from emlab import statics as m


def test_point_charge_gauss_flux_and_potential_gradient():
    from emlab.core import EPS0

    charge, radius = 2e-9, 0.3
    potential, field = m.point_charge([radius, 0, 0], charge)
    assert 4 * np.pi * radius**2 * EPS0 * field[0] == pytest.approx(charge)
    assert potential == pytest.approx(radius * field[0])
    assert np.allclose(field[1:], 0)
    with pytest.raises(ValueError, match="singular"):
        m.point_charge([0, 0, 0], charge)
