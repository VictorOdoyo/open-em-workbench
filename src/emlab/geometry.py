import numpy as np


def rectangle(grid, x_bounds, y_bounds):
    x, y = grid.mesh()
    if x_bounds[0] > x_bounds[1] or y_bounds[0] > y_bounds[1]:
        raise ValueError("ordered region bounds required")
    return (x >= x_bounds[0]) & (x <= x_bounds[1]) & (y >= y_bounds[0]) & (y <= y_bounds[1])


def circle(grid, center, radius):
    if radius <= 0 or not np.isfinite(radius):
        raise ValueError("positive finite radius required")
    x, y = grid.mesh()
    return (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius**2


def circle_distance(grid, center, radius):
    if radius <= 0:
        raise ValueError("positive radius required")
    x, y = grid.mesh()
    return np.hypot(x - center[0], y - center[1]) - radius


def annulus(grid, center, inner_radius, outer_radius):
    if not 0 < inner_radius < outer_radius:
        raise ValueError("ordered positive annular radii required")
    distance = circle_distance(grid, center, outer_radius) + outer_radius
    return (distance >= inner_radius) & (distance <= outer_radius)


def conductor_labels(grid, masks):
    from .grid import boundary_mask

    labels = np.full(grid.shape, -1, dtype=int)
    labels[boundary_mask(grid)] = 0
    for label, mask in enumerate(masks, 1):
        mask = np.asarray(mask)
        if mask.dtype != bool or mask.shape != grid.shape or not np.any(mask):
            raise ValueError("nonempty boolean conductor mask matching grid required")
        if np.any(labels[mask] != -1):
            raise ValueError("conductor overlaps another conductor or the exterior ground")
        labels[mask] = label
    return labels
