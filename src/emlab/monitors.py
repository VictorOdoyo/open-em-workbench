import numpy as np


def fourier_integral(samples, dt_s, frequencies_hz, time_offset_s=0.0, window=None):
    from .core import finite_array

    samples = finite_array(samples)
    frequency = np.atleast_1d(finite_array(frequencies_hz, real=True))
    if samples.ndim not in (1, 2) or len(samples) < 2 or frequency.ndim != 1:
        raise ValueError("one time axis with optional channel axis and one frequency axis required")
    if dt_s <= 0 or not np.isfinite(dt_s + time_offset_s) or np.any(frequency < 0):
        raise ValueError("positive timestep finite origin and nonnegative frequencies required")
    if window is None:
        weights = np.ones(len(samples))
    elif window == "hann":
        weights = np.hanning(len(samples))
    else:
        raise ValueError("window must be None or hann")
    time = time_offset_s + np.arange(len(samples)) * dt_s
    kernel = np.exp(-2j * np.pi * frequency[:, None] * time) * dt_s
    weighted = samples * weights if samples.ndim == 1 else samples * weights[:, None]
    return kernel @ weighted


def phasor(samples, dt_s, frequency_hz, time_offset_s=0.0, window=None):
    samples = np.asarray(samples)
    spectrum = fourier_integral(samples, dt_s, frequency_hz, time_offset_s, window)
    weights = np.ones(len(samples)) if window is None else np.hanning(len(samples))
    scale = 1 if np.iscomplexobj(samples) or frequency_hz == 0 else 2
    return spectrum[0] * scale / (dt_s * weights.sum())


def spectral_ratio(observed, reference, relative_floor=1e-3):
    from .core import finite_array

    observed, reference = finite_array(observed), finite_array(reference)
    if observed.shape != reference.shape or not 0 < relative_floor < 1:
        raise ValueError("aligned spectra and relative floor in (0,1) required")
    peak = float(np.max(abs(reference)))
    if peak == 0:
        raise ValueError("reference contains no incident spectral energy")
    valid = abs(reference) >= peak * relative_floor
    ratio = np.divide(observed, reference, out=np.zeros_like(observed, dtype=complex), where=valid)
    return ratio, valid


def time_gate(samples, dt_s, interval_s, taper_fraction=0.2, time_offset_s=0.0):
    from scipy.signal.windows import tukey

    from .core import finite_array

    values = finite_array(samples)
    if values.ndim not in (1, 2) or dt_s <= 0 or not 0 <= taper_fraction <= 1:
        raise ValueError("time samples positive timestep and valid taper fraction required")
    if len(interval_s) != 2 or interval_s[1] <= interval_s[0]:
        raise ValueError("ordered gate interval required")
    time = time_offset_s + np.arange(len(values)) * dt_s
    selected = (time >= interval_s[0]) & (time <= interval_s[1])
    if selected.sum() < 3:
        raise ValueError("time gate must retain at least three samples")
    weights = np.zeros(len(values))
    weights[selected] = tukey(int(selected.sum()), taper_fraction)
    return values * weights if values.ndim == 1 else values * weights[:, None]


class FourierMonitor:
    def __init__(self, frequencies_hz, dt_s, channels=1):
        from .core import finite_array

        self.frequency = np.atleast_1d(finite_array(frequencies_hz, real=True))
        if (
            self.frequency.ndim != 1
            or np.any(self.frequency < 0)
            or dt_s <= 0
            or not isinstance(channels, int)
            or channels < 1
        ):
            raise ValueError("valid frequencies timestep and channel count required")
        self.dt = dt_s
        self.integral = np.zeros((len(self.frequency), channels), complex)
        self.last_time = None
        self.count = 0

    def add(self, values, time_s, weight=1.0):
        from .core import finite_array

        value = np.atleast_1d(finite_array(values))
        if value.shape != (self.integral.shape[1],) or not np.isfinite(time_s + weight) or weight < 0:
            raise ValueError("matching finite monitor values/time and nonnegative quadrature weight required")
        if self.last_time is not None and abs(time_s - self.last_time - self.dt) > 1e-9 * max(
            self.dt, abs(time_s)
        ):
            raise ValueError("Fourier monitor requires consecutive uniform sample times")
        self.integral += self.dt * weight * np.exp(-2j * np.pi * self.frequency[:, None] * time_s) * value
        self.last_time = time_s
        self.count += 1
