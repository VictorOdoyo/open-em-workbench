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


def material_map(grid, background=1, regions=()):
    from .core import finite_array

    result = np.full(
        grid.shape, background, dtype=np.result_type(background, *[v for _, v in regions], float)
    )
    for mask, value in regions:
        mask = np.asarray(mask)
        if mask.shape != grid.shape or mask.dtype != bool:
            raise ValueError("boolean material masks matching grid required")
        result[mask] = value
    return finite_array(result)


def fill_fraction(grid, predicate, subdivisions=8):
    if not isinstance(subdivisions, int) or not 1 <= subdivisions <= 128:
        raise ValueError("supersampling subdivisions must lie between one and 128")
    x, y = grid.mesh()
    left, right = np.maximum(x - grid.dx / 2, grid.x[0]), np.minimum(x + grid.dx / 2, grid.x[-1])
    bottom, top = np.maximum(y - grid.dy / 2, grid.y[0]), np.minimum(y + grid.dy / 2, grid.y[-1])
    fraction = np.zeros(grid.shape)
    for sy in range(subdivisions):
        yy = bottom + (sy + 0.5) / subdivisions * (top - bottom)
        for sx in range(subdivisions):
            xx = left + (sx + 0.5) / subdivisions * (right - left)
            inside = np.asarray(predicate(xx, yy), dtype=bool)
            if inside.shape != grid.shape:
                raise ValueError("region predicate must return one boolean per sample")
            fraction += inside
    return fraction / subdivisions**2
