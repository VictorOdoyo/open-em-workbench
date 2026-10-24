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
