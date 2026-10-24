import numpy as np


def rlgc(frequency_hz, resistance, inductance, conductance, capacitance):
    from .core import finite_array

    frequency = finite_array(frequency_hz, real=True)
    r, inductance_value, g, c = [
        finite_array(v, real=True) for v in (resistance, inductance, conductance, capacitance)
    ]
    if (
        np.any(frequency <= 0)
        or np.any(r < 0)
        or np.any(g < 0)
        or np.any(inductance_value <= 0)
        or np.any(c <= 0)
    ):
        raise ValueError("positive frequency/L/C and nonnegative R/G required")
    series = r + 2j * np.pi * frequency * inductance_value
    shunt = g + 2j * np.pi * frequency * c
    gamma = np.sqrt(series * shunt)
    impedance = np.sqrt(series / shunt)
    return gamma, impedance
