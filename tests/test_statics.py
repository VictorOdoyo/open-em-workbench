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


def test_dipole_matches_two_opposite_charges_to_second_order_in_separation():
    point = np.array([0.3, 0.2, 0.4])
    moment = np.array([0, 0, 1e-10])
    exact_potential, exact_field = m.electric_dipole(point, moment)
    step = 1e-4
    charge = moment[2] / step
    plus = m.point_charge(point, charge, (0, 0, step / 2))
    minus = m.point_charge(point, -charge, (0, 0, -step / 2))
    assert plus[0] + minus[0] == pytest.approx(exact_potential, rel=1e-7)
    assert np.allclose(plus[1] + minus[1], exact_field, rtol=1e-7)
