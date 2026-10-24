import numpy as np


def parallel_plate(width_m, gap_m, epsilon_r=1):
    from .core import EPS0

    if min(width_m, gap_m, epsilon_r) <= 0:
        raise ValueError("positive plate dimensions and dielectric constant required")
    return EPS0 * epsilon_r * width_m / gap_m


def coaxial(inner_radius_m, outer_radius_m, epsilon_r=1):
    from .core import EPS0

    if not 0 < inner_radius_m < outer_radius_m or epsilon_r <= 0:
        raise ValueError("ordered positive coax radii and dielectric constant required")
    return 2 * np.pi * EPS0 * epsilon_r / np.log(outer_radius_m / inner_radius_m)
