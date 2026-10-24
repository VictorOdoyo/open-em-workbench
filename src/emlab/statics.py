import numpy as np


def point_charge(points_xyz, charge_c, source_xyz=(0.0, 0.0, 0.0), epsilon_r=1.0):
    from .core import EPS0, finite_array

    points, source = finite_array(points_xyz, real=True), finite_array(source_xyz, real=True)
    if points.shape[-1] != 3 or source.shape != (3,) or epsilon_r <= 0 or not np.isfinite(charge_c):
        raise ValueError("three-dimensional geometry finite charge and positive dielectric required")
    displacement = points - source
    radius = np.linalg.norm(displacement, axis=-1)
    if np.any(radius == 0):
        raise ValueError("point-charge observation is singular")
    factor = charge_c / (4 * np.pi * EPS0 * epsilon_r)
    return factor / radius, factor * displacement / radius[..., None] ** 3
