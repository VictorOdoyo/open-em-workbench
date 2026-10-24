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
