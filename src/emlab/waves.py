import numpy as np


def fresnel_normal(impedance_left, impedance_right):
    from .core import finite_array

    left, right = finite_array(impedance_left).astype(complex), finite_array(impedance_right).astype(complex)
    if np.any(left.real <= 0) or np.any(right.real <= 0) or np.any(abs(left + right) < 1e-30):
        raise ValueError("positive-real propagating wave impedances required")
    reflection = (right - left) / (right + left)
    transmission = 1 + reflection
    reflectance = abs(reflection) ** 2
    transmittance = (1 / right).real / (1 / left).real * abs(transmission) ** 2
    return {"r": reflection, "t": transmission, "R": reflectance, "T": transmittance}


def longitudinal_wavenumber(frequency_hz, epsilon_r=1, mu_r=1, transverse_wavenumber=0):
    from .core import C0, finite_array

    frequency = finite_array(frequency_hz, real=True)
    epsilon = finite_array(epsilon_r).astype(complex)
    mu = finite_array(mu_r, real=True)
    transverse = finite_array(transverse_wavenumber, real=True)
    if np.any(frequency <= 0) or np.any(mu <= 0) or np.any(epsilon.imag > 1e-12):
        raise ValueError("positive frequency/mu and passive scalar material required")
    square = (2 * np.pi * frequency / C0) ** 2 * epsilon * mu - transverse**2
    root = np.sqrt(square.astype(complex))
    return np.where(root.imag > 0, -root, root)
