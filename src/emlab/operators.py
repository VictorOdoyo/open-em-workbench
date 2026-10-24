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
