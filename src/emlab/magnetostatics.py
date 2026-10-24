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
