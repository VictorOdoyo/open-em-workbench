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


def electric_dipole(points_xyz, moment_c_m, epsilon_r=1.0):
    from .core import EPS0, finite_array

    points, moment = finite_array(points_xyz, real=True), finite_array(moment_c_m, real=True)
    if points.shape[-1] != 3 or moment.shape != (3,) or epsilon_r <= 0:
        raise ValueError("three-dimensional position/dipole and positive dielectric required")
    radius = np.linalg.norm(points, axis=-1)
    if np.any(radius == 0):
        raise ValueError("dipole observation is singular")
    projection = np.sum(points * moment, axis=-1)
    factor = 1 / (4 * np.pi * EPS0 * epsilon_r)
    potential = factor * projection / radius**3
    field = factor * (
        3 * points * projection[..., None] / radius[..., None] ** 5 - moment / radius[..., None] ** 3
    )
    return potential, field


def grounded_plane(points_xyz, charge_c, height_m, epsilon_r=1.0):
    from .core import finite_array

    points = finite_array(points_xyz, real=True)
    if points.shape[-1] != 3 or height_m <= 0 or np.any(points[..., 2] < 0):
        raise ValueError("positive source height and observations in the upper half space required")
    real = point_charge(points, charge_c, (0, 0, height_m), epsilon_r)
    image = point_charge(points, -charge_c, (0, 0, -height_m), epsilon_r)
    return real[0] + image[0], real[1] + image[1]


def image_surface_charge(radial_distance_m, charge_c, height_m):
    from .core import finite_array

    rho = finite_array(radial_distance_m, real=True)
    if height_m <= 0 or np.any(rho < 0):
        raise ValueError("positive source height and nonnegative radial coordinates required")
    return -charge_c * height_m / (2 * np.pi * (rho**2 + height_m**2) ** 1.5)


def line_charge(points_xy, charge_c_m, source_xy=(0.0, 0.0), epsilon_r=1.0, reference_radius_m=1.0):
    from .core import EPS0, finite_array

    points, source = finite_array(points_xy, real=True), finite_array(source_xy, real=True)
    if points.shape[-1] != 2 or source.shape != (2,) or min(epsilon_r, reference_radius_m) <= 0:
        raise ValueError("two-dimensional points/source and positive dielectric/reference radius required")
    displacement = points - source
    radius = np.linalg.norm(displacement, axis=-1)
    if np.any(radius == 0):
        raise ValueError("line-charge observation is singular")
    factor = charge_c_m / (2 * np.pi * EPS0 * epsilon_r)
    return -factor * np.log(radius / reference_radius_m), factor * displacement / radius[..., None] ** 2
