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
