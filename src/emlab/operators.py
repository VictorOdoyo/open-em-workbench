import numpy as np


def incidence(grid):
    from scipy.sparse import csr_matrix

    nodes = np.arange(grid.nx * grid.ny).reshape(grid.shape)
    tail = np.r_[nodes[:, :-1].ravel(), nodes[:-1, :].ravel()]
    head = np.r_[nodes[:, 1:].ravel(), nodes[1:, :].ravel()]
    rows = np.repeat(np.arange(len(tail)), 2)
    columns = np.column_stack((tail, head)).ravel()
    values = np.tile([-1.0, 1.0], len(tail))
    return csr_matrix((values, (rows, columns)), shape=(len(tail), nodes.size))


def edge_geometry(grid):
    wx = np.full(grid.nx, grid.dx)
    wy = np.full(grid.ny, grid.dy)
    wx[[0, -1]] *= 0.5
    wy[[0, -1]] *= 0.5
    horizontal = grid.ny * (grid.nx - 1)
    vertical = (grid.ny - 1) * grid.nx
    lengths = np.r_[np.full(horizontal, grid.dx), np.full(vertical, grid.dy)]
    areas = np.r_[
        np.broadcast_to(wy[:, None], (grid.ny, grid.nx - 1)).ravel(),
        np.broadcast_to(wx[None, :], (grid.ny - 1, grid.nx)).ravel(),
    ]
    return lengths, areas


def harmonic_edges(grid, values):
    from .core import finite_array

    field = np.broadcast_to(finite_array(values, real=True), grid.shape)
    if np.any(field <= 0):
        raise ValueError("positive scalar material coefficients required")
    x = 2 * field[:, :-1] * field[:, 1:] / (field[:, :-1] + field[:, 1:])
    y = 2 * field[:-1, :] * field[1:, :] / (field[:-1, :] + field[1:, :])
    return np.r_[x.ravel(), y.ravel()]


def stiffness(grid, coefficient=1):
    from scipy.sparse import diags

    derivative = incidence(grid)
    lengths, areas = edge_geometry(grid)
    conductance = harmonic_edges(grid, coefficient) * areas / lengths
    return (derivative.T @ diags(conductance) @ derivative).tocsr()


def eliminate_dirichlet(matrix, rhs, fixed, values=0):
    from .core import finite_array

    fixed = np.asarray(fixed)
    if fixed.dtype != bool or not np.any(fixed) or fixed.size != matrix.shape[0]:
        raise ValueError("nonempty boolean Dirichlet mask matching operator required")
    rhs = finite_array(rhs).reshape(-1)
    if matrix.shape != (fixed.size, fixed.size) or rhs.shape != (fixed.size,):
        raise ValueError("square operator and matching load required")
    specified = np.broadcast_to(finite_array(values), fixed.shape).ravel()
    mask = fixed.ravel()
    free = np.flatnonzero(~mask)
    base = np.zeros(fixed.size, dtype=np.result_type(matrix.dtype, rhs.dtype, specified.dtype))
    base[mask] = specified[mask]
    reduced = matrix[free][:, free].tocsr()
    return reduced, (rhs - matrix @ base)[free], free, base
