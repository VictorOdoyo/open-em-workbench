import numpy as np


def gaussian(time_s, center_s, width_s, frequency_hz=0.0, amplitude=1.0, phase=0.0):
    from .core import finite_array

    time = finite_array(time_s, real=True)
    if (
        width_s <= 0
        or frequency_hz < 0
        or not np.all(np.isfinite([center_s, width_s, frequency_hz, amplitude, phase]))
    ):
        raise ValueError("finite pulse parameters with positive width and nonnegative frequency required")
    relative = time - center_s
    return (
        amplitude
        * np.exp(-0.5 * (relative / width_s) ** 2)
        * np.cos(2 * np.pi * frequency_hz * relative + phase)
    )


def ricker(time_s, center_s, frequency_hz, amplitude=1.0):
    from .core import finite_array

    time = finite_array(time_s, real=True)
    if frequency_hz <= 0 or not np.isfinite(frequency_hz) or not np.isfinite(center_s + amplitude):
        raise ValueError("positive finite pulse frequency and finite center/amplitude required")
    argument = np.pi * frequency_hz * (time - center_s)
    return amplitude * (1 - 2 * argument**2) * np.exp(-(argument**2))
