import numpy as np


def rectangular_cutoff(width_m, height_m, mode_x, mode_y, polarization="te", epsilon_r=1.0, mu_r=1.0):
    from .core import C0

    if (
        min(width_m, height_m, epsilon_r, mu_r) <= 0
        or not isinstance(mode_x, int)
        or not isinstance(mode_y, int)
        or min(mode_x, mode_y) < 0
    ):
        raise ValueError("positive geometry/material and nonnegative integer mode indices required")
    if polarization == "te":
        valid = mode_x + mode_y > 0
    elif polarization == "tm":
        valid = min(mode_x, mode_y) > 0
    else:
        raise ValueError("polarization must be te or tm")
    if not valid:
        raise ValueError("TE00 and TM modes with a zero index are not physical hollow-guide modes")
    return C0 / (2 * np.sqrt(epsilon_r * mu_r)) * np.hypot(mode_x / width_m, mode_y / height_m)
