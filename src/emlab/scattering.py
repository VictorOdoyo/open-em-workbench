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


def widths(coefficients, angles_radians):
    from .core import finite_array

    angles = finite_array(angles_radians, real=True)
    orders = coefficients["orders"]
    values = coefficients["coefficient"]
    k = coefficients["wavenumber_m"]
    pattern = np.sum(values[:, None] * np.exp(1j * orders[:, None] * angles.reshape(1, -1)), axis=0)
    differential = 2 / (np.pi * k) * abs(pattern) ** 2
    scattering = float(4 / k * np.sum(abs(values) ** 2))
    extinction = float(-4 / k * np.sum(values.real))
    return {
        "differential_m_rad": differential.reshape(angles.shape),
        "scattering_m": scattering,
        "extinction_m": extinction,
        "absorption_m": extinction - scattering,
    }


def dielectric_coefficients(frequency_hz, radius_m, epsilon_r, epsilon_background=1.0, maximum_order=None):
    from scipy.special import h2vp, hankel2, jv, jve, jvp

    from .waves import longitudinal_wavenumber

    basis = pec_coefficients(frequency_hz, radius_m, epsilon_background, maximum_order)
    orders = basis["orders"]
    exterior = basis["wavenumber_m"]
    interior = complex(longitudinal_wavenumber(frequency_hz, epsilon_r))
    size_out, size_in = exterior * radius_m, interior * radius_m
    internal = jve(orders, size_in)
    internal_derivative = (jve(orders - 1, size_in) - jve(orders + 1, size_in)) / 2
    outside, outgoing = jv(orders, size_out), hankel2(orders, size_out)
    numerator = interior * internal_derivative * outside - exterior * jvp(orders, size_out) * internal
    denominator = exterior * h2vp(orders, size_out) * internal - interior * internal_derivative * outgoing
    basis["coefficient"] = numerator / denominator
    if not np.all(np.isfinite(basis["coefficient"])):
        raise RuntimeError("cylindrical material response exceeded stable special-function limits")
    return basis


def compare_samples(reference, numerical):
    from .core import relative_error

    return {
        "relative_l2": relative_error(reference, numerical),
        "maximum_absolute": float(np.max(abs(np.asarray(reference) - np.asarray(numerical)))),
    }
