import numpy as np


def courant_step(spacings_m, epsilon_min=1.0, mu_min=1.0, safety=0.95):
    from .core import C0, finite_array

    spacing = finite_array(spacings_m, real=True, ndim=1)
    if spacing.size < 1 or np.any(spacing <= 0) or min(epsilon_min, mu_min) <= 0 or not 0 < safety <= 1:
        raise ValueError("positive spacings/materials and Courant safety in (0,1] required")
    return safety * np.sqrt(epsilon_min * mu_min) / (C0 * np.sqrt(np.sum(1 / spacing**2)))


def electric_coefficients(epsilon_r, conductivity_s_m, dt_s):
    from .core import EPS0, finite_array

    epsilon = EPS0 * finite_array(epsilon_r, real=True)
    conductivity = finite_array(conductivity_s_m, real=True)
    if dt_s <= 0 or not np.isfinite(dt_s) or np.any(epsilon <= 0) or np.any(conductivity < 0):
        raise ValueError("positive permittivity/timestep and nonnegative conductivity required")
    ratio = conductivity * dt_s / (2 * epsilon)
    return (1 - ratio) / (1 + ratio), dt_s / (epsilon * (1 + ratio))


def pml_profile(
    coordinates_m,
    lower_m,
    upper_m,
    thickness_m,
    target_reflection=1e-8,
    order=3,
    kappa_max=3.0,
    alpha_max_s=0.0,
):
    from .core import C0, finite_array

    coordinate = finite_array(coordinates_m, real=True)
    if (
        upper_m <= lower_m
        or thickness_m < 0
        or 2 * thickness_m >= upper_m - lower_m
        or not 0 < target_reflection < 1
        or order < 1
        or kappa_max < 1
        or alpha_max_s < 0
    ):
        raise ValueError("nonoverlapping passive PML geometry and profile parameters required")
    if np.any((coordinate < lower_m) | (coordinate > upper_m)):
        raise ValueError("PML coordinates lie outside its domain")
    if thickness_m == 0:
        return np.zeros_like(coordinate), np.ones_like(coordinate), np.zeros_like(coordinate)
    depth = np.maximum(
        np.maximum(
            (lower_m + thickness_m - coordinate) / thickness_m,
            (coordinate - upper_m + thickness_m) / thickness_m,
        ),
        0,
    )
    sigma_max = -(order + 1) * C0 * np.log(target_reflection) / (2 * thickness_m)
    sigma = sigma_max * depth**order
    kappa = 1 + (kappa_max - 1) * depth**order
    alpha = np.where(depth > 0, alpha_max_s * (1 - depth), 0)
    return sigma, kappa, alpha


def pml_coefficients(sigma_rate, kappa, alpha_rate, dt_s):
    from .core import finite_array

    sigma, kappa, alpha = np.broadcast_arrays(
        finite_array(sigma_rate, real=True),
        finite_array(kappa, real=True),
        finite_array(alpha_rate, real=True),
    )
    if dt_s <= 0 or np.any(sigma < 0) or np.any(kappa < 1) or np.any(alpha < 0):
        raise ValueError("passive CPML rates and positive timestep required")
    decay = np.exp(-(sigma / kappa + alpha) * dt_s)
    denominator = kappa * (sigma + kappa * alpha)
    memory = np.divide(
        sigma * np.expm1(-(sigma / kappa + alpha) * dt_s),
        denominator,
        out=np.zeros_like(sigma),
        where=denominator > 0,
    )
    return decay, memory, kappa


def dispersion_frequency(wavenumber_components, spacings_m, dt_s, speed_m_s):
    from .core import finite_array

    spacing = finite_array(spacings_m, real=True, ndim=1)
    wave = finite_array(wavenumber_components, real=True)
    if wave.shape[-1] != spacing.size or np.any(spacing <= 0) or dt_s <= 0 or speed_m_s <= 0:
        raise ValueError("wave-vector components aligned with positive spacings required")
    if speed_m_s * dt_s * np.sqrt(np.sum(1 / spacing**2)) > 1 + 1e-12:
        raise ValueError("unstable Courant number")
    argument = speed_m_s * dt_s * np.sqrt(np.sum((np.sin(wave * spacing / 2) / spacing) ** 2, axis=-1))
    if np.any(argument > 1 + 1e-12):
        raise ValueError("unresolvable discrete wave")
    return 2 * np.arcsin(np.minimum(argument, 1)) / dt_s
