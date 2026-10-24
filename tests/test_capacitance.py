import numpy as np
import pytest

from emlab import capacitance as m


def test_analytic_capacitances_scale_with_dielectric_and_geometry():
    from emlab.core import EPS0

    assert m.parallel_plate(2, 1, 3) == pytest.approx(6 * EPS0)
    assert m.coaxial(1, np.e, 4) == pytest.approx(8 * np.pi * EPS0)
    assert m.coaxial(2, 6) == pytest.approx(m.coaxial(1, 3))
