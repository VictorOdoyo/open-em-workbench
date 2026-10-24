import numpy as np


def pec_coefficients(frequency_hz, radius_m, epsilon_background=1.0, maximum_order=None):
    from scipy.special import hankel2, jv

    from .core import C0, finite_array

    if min(frequency_hz, radius_m, epsilon_background) <= 0:
        raise ValueError("positive frequency radius and background dielectric required")
    wavenumber = 2 * np.pi * frequency_hz / C0 * np.sqrt(epsilon_background)
    size = wavenumber * radius_m
    order = int(np.ceil(size + 4 * np.cbrt(size) + 8)) if maximum_order is None else maximum_order
    if not isinstance(order, int) or isinstance(order, bool) or not 1 <= order <= 2000:
        raise ValueError("cylindrical harmonic order must lie between one and 2000")
    orders = np.arange(-order, order + 1)
    coefficient = -jv(orders, size) / hankel2(orders, size)
    return {
        "orders": orders,
        "coefficient": finite_array(coefficient),
        "wavenumber_m": wavenumber,
        "radius_m": radius_m,
    }


def exterior_field(coefficients, points_xy, angle_radians=0.0):
    from scipy.special import hankel2

    from .core import finite_array

    points = finite_array(points_xy, real=True)
    if points.shape[-1] != 2:
        raise ValueError("two-dimensional observation coordinates required")
    flat = points.reshape(-1, 2)
    radius = np.linalg.norm(flat, axis=1)
    if np.any(radius < coefficients["radius_m"] * (1 - 1e-12)):
        raise ValueError("exterior cylinder solution is undefined inside the object")
    radius = np.maximum(radius, coefficients["radius_m"])
    angle = np.arctan2(flat[:, 1], flat[:, 0]) - angle_radians
    orders = coefficients["orders"][:, None]
    k = coefficients["wavenumber_m"]
    scattered = np.sum(
        (-1j) ** orders
        * coefficients["coefficient"][:, None]
        * hankel2(orders, k * radius[None, :])
        * np.exp(1j * orders * angle[None, :]),
        axis=0,
    )
    direction = np.array([np.cos(angle_radians), np.sin(angle_radians)])
    incident = np.exp(-1j * k * (flat @ direction))
    shape = points.shape[:-1]
    return {
        "incident": incident.reshape(shape),
        "scattered": scattered.reshape(shape),
        "total": (incident + scattered).reshape(shape),
    }
