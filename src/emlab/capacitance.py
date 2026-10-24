import numpy as np


def parallel_plate(width_m, gap_m, epsilon_r=1):
    from .core import EPS0

    if min(width_m, gap_m, epsilon_r) <= 0:
        raise ValueError("positive plate dimensions and dielectric constant required")
    return EPS0 * epsilon_r * width_m / gap_m


def coaxial(inner_radius_m, outer_radius_m, epsilon_r=1):
    from .core import EPS0

    if not 0 < inner_radius_m < outer_radius_m or epsilon_r <= 0:
        raise ValueError("ordered positive coax radii and dielectric constant required")
    return 2 * np.pi * EPS0 * epsilon_r / np.log(outer_radius_m / inner_radius_m)


def extract(grid, labels, epsilon_r=1):
    from scipy.sparse.linalg import splu

    from .electrostatics import conductor_charges
    from .operators import eliminate_dirichlet, stiffness

    labels = np.asarray(labels)
    if labels.shape != grid.shape or labels.dtype.kind not in "iu" or np.any(labels < -1):
        raise ValueError("integer conductor labels matching grid required")
    identifiers = np.unique(labels[labels >= 0])
    if len(identifiers) < 2 or not np.array_equal(identifiers, np.arange(len(identifiers))):
        raise ValueError("consecutive conductor labels including reference zero required")
    operator = stiffness(grid, epsilon_r)
    fixed = labels >= 0
    reduced, _, free, _ = eliminate_dirichlet(operator, np.zeros(labels.size), fixed)
    factor = splu(reduced.tocsc()) if len(free) else None
    basis = np.zeros((len(identifiers), *grid.shape))
    for label in identifiers[1:]:
        potential = (labels == label).astype(float).ravel()
        rhs = -(operator @ potential)[free]
        if factor is not None:
            potential[free] = factor.solve(rhs)
        basis[label] = potential.reshape(grid.shape)
    basis[0] = 1 - np.sum(basis[1:], axis=0)
    matrix = np.zeros((len(identifiers), len(identifiers)))
    for label in identifiers:
        charges = conductor_charges(grid, labels, basis[label], epsilon_r)
        matrix[:, label] = [charges[int(i)] for i in identifiers]
    return {"matrix": matrix, "basis": basis}


def terminal_response(matrix, voltages):
    from .core import finite_array

    matrix = finite_array(matrix, real=True, ndim=2)
    voltage = finite_array(voltages, real=True, ndim=1)
    scale = max(float(np.max(abs(matrix))), np.finfo(float).tiny)
    if matrix.shape != (len(voltage), len(voltage)) or not np.allclose(matrix, matrix.T, atol=scale * 1e-12):
        raise ValueError("reciprocal square capacitance matrix matching terminal voltages required")
    if np.min(np.linalg.eigvalsh(matrix)) < -scale * 1e-10:
        raise ValueError("passive positive-semidefinite capacitance matrix required")
    charge = matrix @ voltage
    return charge, float(0.5 * voltage @ charge)


def floating_voltages(matrix, driven_indices, driven_voltages, charges=0):
    from .core import finite_array

    matrix = finite_array(matrix, real=True, ndim=2)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("square capacitance matrix required")
    count = len(matrix)
    mvoltage = np.zeros(count)
    terminal_response(matrix, mvoltage)
    driven = np.asarray(driven_indices)
    values = finite_array(driven_voltages, real=True, ndim=1)
    if (
        driven.dtype.kind not in "iu"
        or driven.shape != values.shape
        or driven.size == 0
        or len(np.unique(driven)) != driven.size
        or np.any((driven < 0) | (driven >= count))
    ):
        raise ValueError("unique valid driven terminal indices and matching voltages required")
    specified = np.broadcast_to(finite_array(charges, real=True), (count,))
    floating = np.setdiff1d(np.arange(count), driven)
    mvoltage[driven] = values
    if len(floating):
        block = matrix[np.ix_(floating, floating)]
        if np.linalg.matrix_rank(block) < len(floating):
            raise ValueError("floating conductor system lacks a voltage reference")
        mvoltage[floating] = np.linalg.solve(
            block, specified[floating] - matrix[np.ix_(floating, driven)] @ values
        )
    return mvoltage, matrix @ mvoltage


def transform(matrix, voltage_transform):
    from .core import finite_array

    matrix = finite_array(matrix, real=True, ndim=2)
    transform = finite_array(voltage_transform, real=True, ndim=2)
    if matrix.shape != (transform.shape[0], transform.shape[0]):
        raise ValueError("physical terminal count must match transform rows")
    terminal_response(matrix, np.zeros(len(matrix)))
    return transform.T @ matrix @ transform
