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


def debye(frequency_hz, epsilon_static, epsilon_infinity, relaxation_s):
    from .core import finite_array

    frequency = finite_array(frequency_hz, real=True)
    if (
        np.any(frequency < 0)
        or epsilon_infinity <= 0
        or epsilon_static < epsilon_infinity
        or relaxation_s <= 0
    ):
        raise ValueError("passive Debye parameters and nonnegative frequency required")
    return epsilon_infinity + (epsilon_static - epsilon_infinity) / (
        1 + 2j * np.pi * frequency * relaxation_s
    )


def lorentz(frequency_hz, epsilon_infinity, strength, resonance_hz, damping_hz):
    from .core import finite_array

    f = finite_array(frequency_hz, real=True)
    if np.any(f < 0) or min(epsilon_infinity, resonance_hz, damping_hz) <= 0 or strength < 0:
        raise ValueError("passive Lorentz oscillator parameters required")
    return epsilon_infinity + strength * resonance_hz**2 / (resonance_hz**2 - f**2 + 1j * damping_hz * f)


def drude(frequency_hz, epsilon_infinity, plasma_hz, damping_hz):
    from .core import finite_array

    f = finite_array(frequency_hz, real=True)
    if np.any(f <= 0) or min(epsilon_infinity, plasma_hz, damping_hz) <= 0:
        raise ValueError("positive passive Drude parameters and frequency required")
    return epsilon_infinity - plasma_hz**2 / (f**2 - 1j * damping_hz * f)


def loss_conductivity(frequency_hz, epsilon_r, loss_tangent):
    from .core import EPS0, finite_array

    f = finite_array(frequency_hz, real=True)
    if np.any(f <= 0) or epsilon_r <= 0 or loss_tangent < 0:
        raise ValueError("positive frequency/permittivity and nonnegative loss tangent required")
    return 2 * np.pi * f * EPS0 * epsilon_r * loss_tangent


def skin_depth(frequency_hz, conductivity, mu_r=1):
    from .core import MU0, finite_array

    f = finite_array(frequency_hz, real=True)
    if np.any(f <= 0) or conductivity <= 0 or mu_r <= 0:
        raise ValueError("positive conductor parameters required")
    return np.sqrt(1 / (np.pi * f * MU0 * mu_r * conductivity))


def surface_impedance(frequency_hz, conductivity, mu_r=1):
    return (1 + 1j) / (conductivity * skin_depth(frequency_hz, conductivity, mu_r))


def rotated_tensor(principal_values, rotation):
    from .core import finite_array

    values = finite_array(principal_values, real=True, ndim=1)
    rotation = finite_array(rotation, real=True)
    if rotation.shape != (len(values), len(values)) or np.any(values <= 0):
        raise ValueError("positive principal values and matching rotation matrix required")
    if not np.allclose(rotation.T @ rotation, np.eye(len(values)), atol=1e-12):
        raise ValueError("orthogonal coordinate transformation required")
    return (rotation * values) @ rotation.T


def laminate(permittivities, fractions):
    from .core import finite_array

    epsilon = finite_array(permittivities).astype(complex)
    fraction = finite_array(fractions, real=True, ndim=1)
    if epsilon.shape != fraction.shape or np.any(fraction < 0) or not np.isclose(fraction.sum(), 1):
        raise ValueError("paired material fractions summing to one required")
    if np.any(epsilon.real <= 0) or np.any(epsilon.imag > 1e-12):
        raise ValueError("passive positive-real dielectric constituents required")
    return 1 / np.sum(fraction / epsilon), np.sum(fraction * epsilon)
