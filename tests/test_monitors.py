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


def test_time_gate_preserves_units_and_excludes_outside_events():
    values = np.ones(101)
    gated = m.time_gate(values, 1e-12, (20e-12, 80e-12), 0)
    assert gated[50] == 1 and gated[0] == 0 and gated[-1] == 0
    assert np.max(gated) == 1


def test_streaming_monitor_matches_offline_integral_with_staggered_origin():
    dt = 1e-12
    time = (np.arange(100) + 0.5) * dt
    samples = np.column_stack((np.sin(2 * np.pi * 1e9 * time), np.cos(2 * np.pi * 3e9 * time)))
    frequencies = [1e9, 2e9, 3e9]
    monitor = m.FourierMonitor(frequencies, dt, 2)
    for value, instant in zip(samples, time, strict=True):
        monitor.add(value, instant)
    expected = m.fourier_integral(samples, dt, frequencies, dt / 2)
    assert np.allclose(monitor.integral, expected, rtol=1e-12, atol=1e-25)
    with pytest.raises(ValueError, match="consecutive"):
        monitor.add(samples[-1], time[-1] + 3 * dt)
