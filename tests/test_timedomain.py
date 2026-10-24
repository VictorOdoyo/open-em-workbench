import numpy as np
import pytest

from emlab import timedomain as m


def test_courant_limit_includes_dimension_and_fastest_material():
    from emlab.core import C0

    assert m.courant_step([0.001], safety=1) == pytest.approx(0.001 / C0)
    assert m.courant_step([0.001, 0.001], safety=1) == pytest.approx(0.001 / (C0 * np.sqrt(2)))
    assert m.courant_step([0.001], epsilon_min=4, safety=1) == pytest.approx(0.002 / C0)
