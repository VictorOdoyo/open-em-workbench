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


def fresnel_oblique(frequency_hz, epsilon_left, epsilon_right, angle_radians, polarization="te"):
    from .core import C0, EPS0, MU0

    if not 0 <= angle_radians < np.pi / 2 or epsilon_left <= 0 or epsilon_right <= 0:
        raise ValueError("lossless positive dielectrics and incidence in the forward hemisphere required")
    omega = 2 * np.pi * frequency_hz
    transverse = omega / C0 * np.sqrt(epsilon_left) * np.sin(angle_radians)
    left = longitudinal_wavenumber(frequency_hz, epsilon_left, transverse_wavenumber=transverse)
    right = longitudinal_wavenumber(frequency_hz, epsilon_right, transverse_wavenumber=transverse)
    if polarization == "te":
        yl, yr = left / (omega * MU0), right / (omega * MU0)
    elif polarization == "tm":
        yl, yr = omega * EPS0 * epsilon_left / left, omega * EPS0 * epsilon_right / right
    else:
        raise ValueError("polarization must be te or tm")
    reflection = (yl - yr) / (yl + yr)
    transmission = 1 + reflection
    return {
        "r": reflection,
        "t": transmission,
        "R": abs(reflection) ** 2,
        "T": float(np.real(yr) / np.real(yl) * abs(transmission) ** 2),
        "kz_left": left,
        "kz_right": right,
    }


def critical_angle(index_left, index_right):
    if min(index_left, index_right) <= 0:
        raise ValueError("positive real refractive indices required")
    return None if index_right >= index_left else float(np.arcsin(index_right / index_left))


def brewster_angle(index_left, index_right):
    if min(index_left, index_right) <= 0:
        raise ValueError("positive real refractive indices required")
    return float(np.arctan(index_right / index_left))


def poynting(electric, magnetic):
    from .core import finite_array

    electric, magnetic = finite_array(electric), finite_array(magnetic)
    if electric.shape != magnetic.shape or electric.shape[-1] != 3:
        raise ValueError("matching three-component field vectors required")
    return 0.5 * np.real(np.cross(electric, magnetic.conj()))
