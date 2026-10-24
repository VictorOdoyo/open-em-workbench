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
