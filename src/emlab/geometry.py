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
