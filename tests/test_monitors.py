import numpy as np
import pytest

from emlab import monitors as m


def test_fourier_integral_uses_actual_half_step_time_origin():
    dt, count, frequency, phase = 1e-12, 1000, 1e10, 0.4
    time = (np.arange(count) + 0.5) * dt
    signal = np.cos(2 * np.pi * frequency * time + phase)
    value = m.fourier_integral(signal, dt, [frequency], dt / 2)[0]
    assert value == pytest.approx(count * dt / 2 * np.exp(1j * phase), rel=1e-12)
