import numpy as np
import pytest

from emlab import sources as m


def test_gaussian_pulse_has_known_center_width_and_carrier_phase():
    assert m.gaussian(2e-9, 2e-9, 0.1e-9, 1e9, 3) == pytest.approx(3)
    assert m.gaussian(2e-9 + 0.1e-9, 2e-9, 0.1e-9) == pytest.approx(np.exp(-0.5))
    assert m.gaussian(2e-9, 2e-9, 0.1e-9, phase=np.pi / 2) == pytest.approx(0, abs=1e-14)
