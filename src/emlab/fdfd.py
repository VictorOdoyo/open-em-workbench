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


def operator(grid, frequency_hz, epsilon_r=1.0, pml_cells=12):
    from scipy.sparse import diags

    from .core import C0, finite_array
    from .grid import dual_weights
    from .operators import edge_geometry, incidence

    epsilon = np.broadcast_to(finite_array(epsilon_r), grid.shape).astype(complex)
    if np.any(epsilon.imag > 1e-12):
        raise ValueError("passive permittivity required under exp(+i omega t)")
    stretch = coordinate_stretch(grid, frequency_hz, pml_cells)
    horizontal = stretch["ye"][:, None] / stretch["xh"][None, :]
    vertical = stretch["xe"][None, :] / stretch["yh"][:, None]
    length, area = edge_geometry(grid)
    coefficient = np.r_[horizontal.ravel(), vertical.ravel()] * area / length
    derivative = incidence(grid)
    stiffness = derivative.T @ diags(coefficient) @ derivative
    volume = dual_weights(grid) * stretch["ye"][:, None] * stretch["xe"][None, :]
    matrix = (stiffness - diags(((2 * np.pi * frequency_hz / C0) ** 2 * epsilon * volume).ravel())).tocsr()
    return {"matrix": matrix, "coordinate_volume": volume, "stretch": stretch}


def solve(grid, frequency_hz, current_density, epsilon_r=1.0, pml_cells=12, pec_mask=None):
    from .core import MU0, finite_array
    from .grid import boundary_mask
    from .operators import eliminate_dirichlet, solve_linear

    model = operator(grid, frequency_hz, epsilon_r, pml_cells)
    fixed = boundary_mask(grid)
    if pec_mask is not None:
        mask = np.asarray(pec_mask)
        if mask.shape != grid.shape or mask.dtype != bool:
            raise ValueError("boolean PEC mask matching the electric grid required")
        fixed |= mask
    current = np.broadcast_to(finite_array(current_density), grid.shape)
    if np.any(current[fixed] != 0):
        raise ValueError("impressed current cannot occupy PEC nodes")
    load = (-2j * np.pi * frequency_hz * MU0 * current * model["coordinate_volume"]).ravel()
    reduced, rhs, free, field = eliminate_dirichlet(model["matrix"], load, fixed)
    solution, residual, _ = solve_linear(reduced, rhs)
    field[free] = solution
    return {"electric": field.reshape(grid.shape), "residual": residual}
