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
