import numpy as np
import pytest

from emlab import monitors as m


def test_fourier_integral_uses_actual_half_step_time_origin():
    dt, count, frequency, phase = 1e-12, 1000, 1e10, 0.4
    time = (np.arange(count) + 0.5) * dt
    signal = np.cos(2 * np.pi * frequency * time + phase)
    value = m.fourier_integral(signal, dt, [frequency], dt / 2)[0]
    assert value == pytest.approx(count * dt / 2 * np.exp(1j * phase), rel=1e-12)


def test_phasor_extraction_recovers_peak_not_rms_amplitude():
    dt, count, frequency = 1e-12, 1000, 1e10
    time = np.arange(count) * dt
    complex_signal = 3 * np.exp(1j * (2 * np.pi * frequency * time + 0.2))
    real_signal = complex_signal.real
    assert m.phasor(real_signal, dt, frequency) == pytest.approx(3 * np.exp(0.2j))
    assert m.phasor(complex_signal, dt, frequency) == pytest.approx(3 * np.exp(0.2j))


def test_spectral_normalization_retains_an_explicit_validity_mask():
    reference = np.array([0, 1e-8, 1 + 1j])
    observed = reference * (0.2 + 0.4j)
    ratio, valid = m.spectral_ratio(observed, reference)
    assert np.array_equal(valid, [False, False, True])
    assert ratio[-1] == pytest.approx(0.2 + 0.4j)
    with pytest.raises(ValueError, match="no incident"):
        m.spectral_ratio(np.ones(3), np.zeros(3))
