import numpy as np


def permittivity(frequency_hz, epsilon_r=1, conductivity=0):
    from .core import EPS0, finite_array

    frequency = finite_array(frequency_hz, real=True)
    epsilon_r = finite_array(epsilon_r, real=True)
    conductivity = finite_array(conductivity, real=True)
    if np.any(frequency <= 0) or np.any(epsilon_r <= 0) or np.any(conductivity < 0):
        raise ValueError("positive frequency/permittivity and nonnegative conductivity required")
    return epsilon_r - 1j * conductivity / (2 * np.pi * frequency * EPS0)


def propagation(frequency_hz, epsilon_r=1, mu_r=1):
    from .core import C0, ETA0, finite_array

    frequency = finite_array(frequency_hz, real=True)
    epsilon = finite_array(epsilon_r).astype(complex)
    mu = finite_array(mu_r, real=True)
    if np.any(frequency <= 0) or np.any(mu <= 0) or np.any(epsilon.imag > 1e-12) or np.any(abs(epsilon) == 0):
        raise ValueError("positive frequency/mu and nonzero passive permittivity required")
    index = np.sqrt(mu * epsilon)
    index = np.where(index.imag > 0, -index, index)
    gamma = 1j * (2 * np.pi * frequency / C0) * index
    impedance = ETA0 * mu / index
    return gamma, impedance
