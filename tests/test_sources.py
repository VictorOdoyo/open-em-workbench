import numpy as np
import pytest

from emlab import sources as m


def test_gaussian_pulse_has_known_center_width_and_carrier_phase():
    assert m.gaussian(2e-9, 2e-9, 0.1e-9, 1e9, 3) == pytest.approx(3)
    assert m.gaussian(2e-9 + 0.1e-9, 2e-9, 0.1e-9) == pytest.approx(np.exp(-0.5))
    assert m.gaussian(2e-9, 2e-9, 0.1e-9, phase=np.pi / 2) == pytest.approx(0, abs=1e-14)


def test_ricker_pulse_is_broadband_zero_mean_with_unit_center():
    time = np.linspace(-10e-9, 10e-9, 10001)
    pulse = m.ricker(time, 0, 1e9)
    assert pulse[5000] == 1
    assert abs(np.trapezoid(pulse, time)) < 1e-24


def test_continuous_source_is_causal_and_reaches_requested_amplitude():
    assert m.continuous(-1, 1, 1) == 0
    assert m.continuous(0, 1, 1) == 0
    assert m.continuous(0.5, 1, 1) == pytest.approx(-0.5)
    assert m.continuous(2, 1, 1) == pytest.approx(1)
