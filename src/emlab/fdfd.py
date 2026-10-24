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


def green_2d(frequency_hz, radius_m, epsilon_r=1.0):
    from scipy.special import hankel2

    from .core import C0, finite_array

    radius = finite_array(radius_m, real=True)
    if min(frequency_hz, epsilon_r) <= 0 or np.any(radius <= 0):
        raise ValueError("positive frequency/material and nonsingular observation radius required")
    return -0.25j * hankel2(0, 2 * np.pi * frequency_hz / C0 * np.sqrt(epsilon_r) * radius)


def plane_wave(grid, frequency_hz, epsilon_background=1.0, angle_radians=0.0):
    from scipy.optimize import brentq

    from .core import C0

    if min(frequency_hz, epsilon_background) <= 0 or not np.isfinite(angle_radians):
        raise ValueError("positive background frequency/dielectric and finite incidence angle required")
    direction = np.array([np.cos(angle_radians), np.sin(angle_radians)])
    spacing = np.array([grid.dx, grid.dy])
    target = (2 * np.pi * frequency_hz / C0) ** 2 * epsilon_background
    limit = min(np.pi / (spacing[abs(direction) > 1e-14] * abs(direction[abs(direction) > 1e-14])))

    def relation(value):
        return float(np.sum(4 * np.sin(value * direction * spacing / 2) ** 2 / spacing**2) - target)

    if relation(limit) <= 0:
        raise ValueError("incident frequency lies outside the discrete propagation band")
    magnitude = brentq(relation, 0, limit, xtol=1e-13, rtol=1e-14)
    wavevector = magnitude * direction
    x, y = grid.mesh()
    return np.exp(-1j * (wavevector[0] * x + wavevector[1] * y)), wavevector


def magnetic_fields(grid, frequency_hz, electric, pml_cells=12):
    from .core import MU0, finite_array
    from .operators import gradient_faces

    electric = finite_array(electric)
    if electric.shape != grid.shape:
        raise ValueError("electric phasor must match the grid")
    stretch = coordinate_stretch(grid, frequency_hz, pml_cells)
    dx, dy = gradient_faces(grid, electric)
    omega_mu = 2 * np.pi * frequency_hz * MU0
    hx = 1j * dy / (omega_mu * stretch["yh"][:, None])
    hy = -1j * dx / (omega_mu * stretch["xh"][None, :])
    return hx, hy


def scatter(
    grid, frequency_hz, epsilon_r=1.0, pec_mask=None, pml_cells=12, epsilon_background=1.0, angle_radians=0.0
):
    from .core import C0, finite_array
    from .grid import boundary_mask
    from .operators import eliminate_dirichlet, solve_linear

    model = operator(grid, frequency_hz, epsilon_r, pml_cells)
    epsilon = np.broadcast_to(finite_array(epsilon_r), grid.shape)
    incident, wavevector = plane_wave(grid, frequency_hz, epsilon_background, angle_radians)
    obstacle = np.zeros(grid.shape, bool) if pec_mask is None else np.asarray(pec_mask)
    if obstacle.shape != grid.shape or obstacle.dtype != bool:
        raise ValueError("boolean PEC geometry matching grid required")
    margin = pml_cells + 1
    allowed = np.zeros(grid.shape, bool)
    allowed[margin:-margin, margin:-margin] = True
    if np.any(obstacle & ~allowed) or np.any(abs(epsilon[~allowed] - epsilon_background) > 1e-12):
        raise ValueError("all scattering contrast must lie outside the PML and exterior boundary")
    fixed = boundary_mask(grid) | obstacle
    boundary_value = np.where(obstacle, -incident, 0)
    load = (
        (2 * np.pi * frequency_hz / C0) ** 2
        * (epsilon - epsilon_background)
        * model["coordinate_volume"]
        * incident
    )
    reduced, rhs, free, scattered = eliminate_dirichlet(model["matrix"], load.ravel(), fixed, boundary_value)
    solution, residual, _ = solve_linear(reduced, rhs)
    scattered[free] = solution
    scattered = scattered.reshape(grid.shape)
    return {
        "incident": incident,
        "scattered": scattered,
        "total": incident + scattered,
        "wavevector_m": wavevector,
        "residual": residual,
    }
