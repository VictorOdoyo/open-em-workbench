import numpy as np


def segment_field(points_xyz, start_xyz, end_xyz, current_a=1.0):
    from .core import MU0, finite_array

    points = finite_array(points_xyz, real=True)
    start, end = finite_array(start_xyz, real=True), finite_array(end_xyz, real=True)
    if points.shape[-1] != 3 or start.shape != (3,) or end.shape != (3,) or not np.isfinite(current_a):
        raise ValueError("three-dimensional observation points and finite segment current required")
    wire = end - start
    length = float(np.linalg.norm(wire))
    if length == 0:
        raise ValueError("nonzero filament length required")
    r1, r2 = points - start, points - end
    n1, n2 = np.linalg.norm(r1, axis=-1), np.linalg.norm(r2, axis=-1)
    cross = np.cross(r1, r2)
    squared = np.sum(cross**2, axis=-1)
    tolerance = 32 * np.finfo(float).eps * np.maximum(length, np.maximum(n1, n2))
    distance = np.sqrt(squared) / length
    projection = np.sum(r1 * wire, axis=-1) / length
    singular = (distance <= tolerance) & (projection >= -tolerance) & (projection <= length + tolerance)
    if np.any(singular) or np.any(n1 == 0) or np.any(n2 == 0):
        raise ValueError("observation lies on a singular current filament")
    factor = np.sum(wire * r1, axis=-1) / n1 - np.sum(wire * r2, axis=-1) / n2
    coefficient = np.divide(factor, squared, out=np.zeros_like(factor), where=distance > tolerance)
    return MU0 * current_a / (4 * np.pi) * cross * coefficient[..., None]


def circular_loop(radius_m, segments=128, center_xyz=(0.0, 0.0, 0.0)):
    from .core import finite_array

    center = finite_array(center_xyz, real=True)
    if radius_m <= 0 or center.shape != (3,) or not isinstance(segments, int) or segments < 4:
        raise ValueError("positive radius three-dimensional center and at least four segments required")
    angle = np.arange(segments) * 2 * np.pi / segments
    vertices = center + radius_m * np.column_stack((np.cos(angle), np.sin(angle), np.zeros(segments)))
    return np.vstack((vertices, vertices[0]))


def polyline_field(points_xyz, vertices_xyz, current_a=1.0):
    from itertools import pairwise

    from .core import finite_array

    points = finite_array(points_xyz, real=True)
    vertices = finite_array(vertices_xyz, real=True, ndim=2)
    if points.shape[-1] != 3 or vertices.shape[1] != 3 or len(vertices) < 2:
        raise ValueError("three-dimensional observation and filament vertices required")
    result = np.zeros_like(points, dtype=float)
    for start, end in pairwise(vertices):
        result += segment_field(points, start, end, current_a)
    return result


def loop_axis(radius_m, axial_position_m, current_a=1.0):
    from .core import MU0, finite_array

    z = finite_array(axial_position_m, real=True)
    if radius_m <= 0 or not np.isfinite(radius_m + current_a):
        raise ValueError("positive finite loop radius and finite current required")
    return MU0 * current_a * radius_m**2 / (2 * (radius_m**2 + z**2) ** 1.5)


def circular_field(points_xyz, radius_m, current_a=1.0):
    from scipy.special import ellipe, ellipk

    from .core import MU0, finite_array

    points = finite_array(points_xyz, real=True)
    if points.shape[-1] != 3 or radius_m <= 0 or not np.isfinite(radius_m + current_a):
        raise ValueError("three-dimensional points and positive finite coil radius required")
    flat = points.reshape(-1, 3)
    rho, z = np.hypot(flat[:, 0], flat[:, 1]), flat[:, 2]
    axis = rho == 0
    denominator = (radius_m - rho) ** 2 + z**2
    if np.any(denominator == 0):
        raise ValueError("observation lies on a singular circular filament")
    scale = np.sqrt((radius_m + rho) ** 2 + z**2)
    parameter = 4 * radius_m * rho / scale**2
    first, second = ellipk(parameter), ellipe(parameter)
    radial = np.zeros_like(rho)
    np.divide(
        MU0 * current_a * z * (-first + (radius_m**2 + rho**2 + z**2) / denominator * second),
        2 * np.pi * rho * scale,
        out=radial,
        where=~axis,
    )
    axial = (
        MU0 * current_a / (2 * np.pi * scale) * (first + (radius_m**2 - rho**2 - z**2) / denominator * second)
    )
    axial[axis] = loop_axis(radius_m, z[axis], current_a)
    direction_x = np.divide(flat[:, 0], rho, out=np.zeros_like(rho), where=~axis)
    direction_y = np.divide(flat[:, 1], rho, out=np.zeros_like(rho), where=~axis)
    return np.column_stack((radial * direction_x, radial * direction_y, axial)).reshape(points.shape)


def segment_potential(points_xyz, start_xyz, end_xyz, current_a=1.0):
    from .core import MU0, finite_array

    points = finite_array(points_xyz, real=True)
    start, end = finite_array(start_xyz, real=True), finite_array(end_xyz, real=True)
    if points.shape[-1] != 3 or start.shape != (3,) or end.shape != (3,):
        raise ValueError("three-dimensional points and segment endpoints required")
    wire = end - start
    length = float(np.linalg.norm(wire))
    if length == 0:
        raise ValueError("nonzero filament length required")
    distance = np.linalg.norm(points - start, axis=-1) + np.linalg.norm(points - end, axis=-1)
    if np.any(distance <= length * (1 + 1e-14)):
        raise ValueError("singular filament observation")
    integral = np.log1p(2 * length / (distance - length))
    return MU0 * current_a / (4 * np.pi) * integral[..., None] * wire / length


def mutual_inductance(first_vertices, second_vertices, quadrature_order=4):
    from numpy.polynomial.legendre import leggauss

    from .core import MU0, finite_array

    def samples(vertices):
        vertices = finite_array(vertices, real=True, ndim=2)
        if (
            vertices.shape[1] != 3
            or len(vertices) < 4
            or not np.allclose(vertices[0], vertices[-1], rtol=0, atol=1e-14)
        ):
            raise ValueError("closed three-dimensional polygon filaments required")
        difference = np.diff(vertices, axis=0)
        if np.any(np.linalg.norm(difference, axis=1) == 0):
            raise ValueError("zero-length filament edge")
        nodes, weights = leggauss(quadrature_order)
        position = vertices[:-1, None, :] + (nodes[None, :, None] + 1) / 2 * difference[:, None, :]
        differential = weights[None, :, None] / 2 * difference[:, None, :]
        return position.reshape(-1, 3), differential.reshape(-1, 3)

    if not isinstance(quadrature_order, int) or not 1 <= quadrature_order <= 16:
        raise ValueError("quadrature order between one and sixteen required")
    first, dl_first = samples(first_vertices)
    second, dl_second = samples(second_vertices)
    distance = np.linalg.norm(first[:, None, :] - second[None, :, :], axis=-1)
    if np.any(distance < 1e-14):
        raise ValueError("intersecting or coincident filaments require a finite-wire self model")
    return float(MU0 / (4 * np.pi) * np.sum((dl_first @ dl_second.T) / distance))


def coaxial_mutual(radius_first_m, radius_second_m, separation_m):
    from scipy.special import ellipe, ellipk

    from .core import MU0

    if min(radius_first_m, radius_second_m) <= 0 or not np.isfinite(separation_m):
        raise ValueError("positive loop radii and finite separation required")
    a, b, z = radius_first_m, radius_second_m, separation_m
    denominator = (a + b) ** 2 + z**2
    parameter = 4 * a * b / denominator
    if parameter >= 1:
        raise ValueError("coincident circular filaments have singular mutual inductance")
    if parameter < 1e-6:
        return MU0 * np.pi * a * a * b * b / (2 * denominator**1.5) * (1 + 0.75 * parameter)
    return (
        MU0
        * np.sqrt(a * b)
        * ((2 - parameter) * ellipk(parameter) - 2 * ellipe(parameter))
        / np.sqrt(parameter)
    )
