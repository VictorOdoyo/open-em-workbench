import numpy as np


def coordinate_stretch(grid, frequency_hz, pml_cells=12, target_reflection=1e-8):
    from .timedomain import pml_profile

    if (
        frequency_hz <= 0
        or not np.isfinite(frequency_hz)
        or not isinstance(pml_cells, int)
        or isinstance(pml_cells, bool)
        or pml_cells < 0
        or 2 * pml_cells >= min(grid.nx, grid.ny) - 1
    ):
        raise ValueError("positive frequency and nonoverlapping integer PML thickness required")
    result = {}
    for name, coordinate, spacing in (("x", grid.x, grid.dx), ("y", grid.y, grid.dy)):
        for suffix, locations in (("e", coordinate), ("h", (coordinate[:-1] + coordinate[1:]) / 2)):
            sigma, kappa, alpha = pml_profile(
                locations,
                coordinate[0],
                coordinate[-1],
                pml_cells * spacing,
                target_reflection,
                kappa_max=1.0,
            )
            result[name + suffix] = kappa + sigma / (alpha + 2j * np.pi * frequency_hz)
    return result
