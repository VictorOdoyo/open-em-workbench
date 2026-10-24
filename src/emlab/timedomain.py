import numpy as np


def courant_step(spacings_m, epsilon_min=1.0, mu_min=1.0, safety=0.95):
    from .core import C0, finite_array

    spacing = finite_array(spacings_m, real=True, ndim=1)
    if spacing.size < 1 or np.any(spacing <= 0) or min(epsilon_min, mu_min) <= 0 or not 0 < safety <= 1:
        raise ValueError("positive spacings/materials and Courant safety in (0,1] required")
    return safety * np.sqrt(epsilon_min * mu_min) / (C0 * np.sqrt(np.sum(1 / spacing**2)))
