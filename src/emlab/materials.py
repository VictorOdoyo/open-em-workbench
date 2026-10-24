import numpy as np


def permittivity(frequency_hz, epsilon_r=1, conductivity=0):
    from .core import EPS0, finite_array

    frequency = finite_array(frequency_hz, real=True)
    epsilon_r = finite_array(epsilon_r, real=True)
    conductivity = finite_array(conductivity, real=True)
    if np.any(frequency <= 0) or np.any(epsilon_r <= 0) or np.any(conductivity < 0):
        raise ValueError("positive frequency/permittivity and nonnegative conductivity required")
    return epsilon_r - 1j * conductivity / (2 * np.pi * frequency * EPS0)
