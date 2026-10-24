import numpy as np
import pytest

from emlab import timedomain as m


def test_courant_limit_includes_dimension_and_fastest_material():
    from emlab.core import C0

    assert m.courant_step([0.001], safety=1) == pytest.approx(0.001 / C0)
    assert m.courant_step([0.001, 0.001], safety=1) == pytest.approx(0.001 / (C0 * np.sqrt(2)))
    assert m.courant_step([0.001], epsilon_min=4, safety=1) == pytest.approx(0.002 / C0)


def test_conductive_update_matches_trapezoidal_ohmic_decay():
    from emlab.core import EPS0

    dt, conductivity = 1e-12, 0.1
    a, b = m.electric_coefficients(3, conductivity, dt)
    assert abs(a) <= 1
    assert a == pytest.approx((3 * EPS0 - conductivity * dt / 2) / (3 * EPS0 + conductivity * dt / 2))
    assert b == pytest.approx(dt / (3 * EPS0 + conductivity * dt / 2))
    assert m.electric_coefficients(1, 0, dt)[0] == 1
