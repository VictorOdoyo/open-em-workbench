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


def work_balance(grid, potential, epsilon_r, fixed, charge_density=0):
    from .core import finite_array
    from .grid import dual_weights

    potential = finite_array(potential, real=True)
    fixed = np.asarray(fixed)
    if fixed.shape != grid.shape or fixed.dtype != bool:
        raise ValueError("boolean Dirichlet mask required")
    density = np.broadcast_to(finite_array(charge_density, real=True), grid.shape)
    reaction = charge_reaction(grid, potential, epsilon_r, density)
    stored = energy(grid, potential, epsilon_r)
    source_work = 0.5 * float(np.sum(density * dual_weights(grid) * potential))
    boundary_work = 0.5 * float(np.sum(reaction[fixed] * potential[fixed]))
    return {
        "field_energy": stored,
        "source_work": source_work,
        "boundary_work": boundary_work,
        "defect": stored - source_work - boundary_work,
    }


def displacement(grid, potential, epsilon_r=1):
    from .core import EPS0
    from .operators import harmonic_edges

    ex, ey = electric_field(grid, potential)
    edges = harmonic_edges(grid, epsilon_r)
    split = ex.size
    return EPS0 * edges[:split].reshape(ex.shape) * ex, EPS0 * edges[split:].reshape(ey.shape) * ey


def energy_sensitivity(grid, potential, epsilon_r):
    from .core import EPS0, finite_array
    from .operators import edge_geometry, incidence

    epsilon = np.broadcast_to(finite_array(epsilon_r, real=True), grid.shape)
    potential = finite_array(potential, real=True)
    if potential.shape != grid.shape or np.any(epsilon <= 0):
        raise ValueError("matching potential and positive dielectric map required")
    nodes = np.arange(grid.nx * grid.ny).reshape(grid.shape)
    tails = np.r_[nodes[:, :-1].ravel(), nodes[:-1].ravel()]
    heads = np.r_[nodes[:, 1:].ravel(), nodes[1:].ravel()]
    a, b = epsilon.ravel()[tails], epsilon.ravel()[heads]
    lengths, areas = edge_geometry(grid)
    difference = incidence(grid) @ potential.ravel()
    factor = 0.5 * EPS0 * areas / lengths * difference**2
    derivative = np.zeros(epsilon.size)
    np.add.at(derivative, tails, factor * 2 * b * b / (a + b) ** 2)
    np.add.at(derivative, heads, factor * 2 * a * a / (a + b) ** 2)
    return derivative.reshape(grid.shape)


def maxwell_stress(electric_x, electric_y, epsilon_r=1.0):
    from .core import EPS0, finite_array

    ex, ey = np.broadcast_arrays(finite_array(electric_x, real=True), finite_array(electric_y, real=True))
    if not np.isscalar(epsilon_r) or not np.isfinite(epsilon_r) or epsilon_r <= 0:
        raise ValueError("one positive dielectric constant for the homogeneous contour medium required")
    result = np.empty((*ex.shape, 2, 2))
    result[..., 0, 0] = 0.5 * EPS0 * epsilon_r * (ex**2 - ey**2)
    result[..., 1, 1] = -result[..., 0, 0]
    result[..., 0, 1] = result[..., 1, 0] = EPS0 * epsilon_r * ex * ey
    return result


def rectangular_force(grid, stress, index_bounds):
    from .core import finite_array

    tensor = finite_array(stress, real=True)
    if tensor.shape != (*grid.shape, 2, 2) or len(index_bounds) != 4:
        raise ValueError("stress tensor per node and y0/y1/x0/x1 index bounds required")
    y0, y1, x0, x1 = index_bounds
    if not 0 <= y0 < y1 < grid.ny or not 0 <= x0 < x1 < grid.nx:
        raise ValueError("ordered contour indices inside the grid required")
    right = np.trapezoid(tensor[y0 : y1 + 1, x1, :, 0], grid.y[y0 : y1 + 1], axis=0)
    left = np.trapezoid(tensor[y0 : y1 + 1, x0, :, 0], grid.y[y0 : y1 + 1], axis=0)
    top = np.trapezoid(tensor[y1, x0 : x1 + 1, :, 1], grid.x[x0 : x1 + 1], axis=0)
    bottom = np.trapezoid(tensor[y0, x0 : x1 + 1, :, 1], grid.x[x0 : x1 + 1], axis=0)
    return right - left + top - bottom
