import numpy as np


def solve(grid, epsilon_r=1, fixed=None, voltage=0, charge_density=0, method="direct"):
    from .core import EPS0, finite_array
    from .grid import boundary_mask, dual_weights
    from .operators import eliminate_dirichlet, solve_linear, stiffness

    fixed = boundary_mask(grid) if fixed is None else np.asarray(fixed)
    if fixed.shape != grid.shape or fixed.dtype != bool:
        raise ValueError("boolean fixed-potential mask matching grid required")
    voltage = np.broadcast_to(finite_array(voltage, real=True), grid.shape)
    density = np.broadcast_to(finite_array(charge_density, real=True), grid.shape)
    operator = stiffness(grid, epsilon_r)
    load = (density * dual_weights(grid) / EPS0).ravel()
    reduced, rhs, free, potential = eliminate_dirichlet(operator, load, fixed, voltage)
    solution, residual, iterations = solve_linear(reduced, rhs, method)
    potential[free] = solution
    return {"potential": potential.reshape(grid.shape), "residual": residual, "iterations": iterations}


def electric_field(grid, potential):
    from .operators import gradient_faces

    gx, gy = gradient_faces(grid, potential)
    return -gx, -gy


def nodal_field(grid, potential):
    from .core import finite_array

    potential = finite_array(potential, real=True)
    if potential.shape != grid.shape:
        raise ValueError("potential must match grid")
    gy, gx = np.gradient(potential, grid.dy, grid.dx, edge_order=2)
    return -gx, -gy


def charge_reaction(grid, potential, epsilon_r=1, charge_density=0):
    from .core import EPS0, finite_array
    from .grid import dual_weights
    from .operators import stiffness

    potential = finite_array(potential, real=True)
    if potential.shape != grid.shape:
        raise ValueError("potential must match grid")
    density = np.broadcast_to(finite_array(charge_density, real=True), grid.shape)
    return (EPS0 * (stiffness(grid, epsilon_r) @ potential.ravel())).reshape(
        grid.shape
    ) - density * dual_weights(grid)


def energy(grid, potential, epsilon_r=1):
    from .core import EPS0, finite_array
    from .operators import stiffness

    potential = finite_array(potential, real=True)
    if potential.shape != grid.shape:
        raise ValueError("potential must match grid")
    values = potential.ravel()
    return float(0.5 * EPS0 * (values @ (stiffness(grid, epsilon_r) @ values)))


def conductor_charges(grid, labels, potential, epsilon_r=1, charge_density=0):
    labels = np.asarray(labels)
    potential = np.asarray(potential)
    if labels.shape != grid.shape or labels.dtype.kind not in "iu" or np.any(labels < -1):
        raise ValueError("integer conductor labels with -1 for free nodes required")
    if potential.shape != grid.shape:
        raise ValueError("potential must match grid")
    reaction = charge_reaction(grid, potential, epsilon_r, charge_density)
    charges = {}
    for label in np.unique(labels[labels >= 0]):
        mask = labels == label
        if np.ptp(potential[mask]) > 1e-9 * max(1, float(np.max(abs(potential[mask])))):
            raise ValueError("each conductor must be equipotential")
        charges[int(label)] = float(np.sum(reaction[mask]))
    return charges


def manufactured_sine(grid, epsilon_r=1):
    from .core import EPS0

    x, y = grid.mesh()
    width, height = grid.x[-1] - grid.x[0], grid.y[-1] - grid.y[0]
    potential = np.sin(np.pi * (x - grid.x0) / width) * np.sin(np.pi * (y - grid.y0) / height)
    density = EPS0 * epsilon_r * np.pi**2 * (1 / width**2 + 1 / height**2) * potential
    return potential, density
