import numpy as np


def rectangular_cutoff(width_m, height_m, mode_x, mode_y, polarization="te", epsilon_r=1.0, mu_r=1.0):
    from .core import C0

    if (
        min(width_m, height_m, epsilon_r, mu_r) <= 0
        or not isinstance(mode_x, int)
        or not isinstance(mode_y, int)
        or min(mode_x, mode_y) < 0
    ):
        raise ValueError("positive geometry/material and nonnegative integer mode indices required")
    if polarization == "te":
        valid = mode_x + mode_y > 0
    elif polarization == "tm":
        valid = min(mode_x, mode_y) > 0
    else:
        raise ValueError("polarization must be te or tm")
    if not valid:
        raise ValueError("TE00 and TM modes with a zero index are not physical hollow-guide modes")
    return C0 / (2 * np.sqrt(epsilon_r * mu_r)) * np.hypot(mode_x / width_m, mode_y / height_m)


def cavity_modes(grid, epsilon_r=1.0, mu_r=1.0, polarization="tm", count=6):
    from scipy.sparse import diags
    from scipy.sparse.linalg import eigsh

    from .core import C0, finite_array
    from .grid import boundary_mask, dual_weights
    from .operators import stiffness

    epsilon = np.broadcast_to(finite_array(epsilon_r, real=True), grid.shape)
    mu = np.broadcast_to(finite_array(mu_r, real=True), grid.shape)
    if np.any(epsilon <= 0) or np.any(mu <= 0) or not isinstance(count, int) or count < 1:
        raise ValueError("positive material maps and mode count required")
    if polarization == "tm":
        operator = stiffness(grid, 1 / mu)
        mass = (epsilon * dual_weights(grid)).ravel()
        free = np.flatnonzero(~boundary_mask(grid).ravel())
        skip = 0
    elif polarization == "te":
        operator = stiffness(grid, 1 / epsilon)
        mass = (mu * dual_weights(grid)).ravel()
        free = np.arange(epsilon.size)
        skip = 1
    else:
        raise ValueError("polarization must be tm or te")
    if count + skip >= len(free):
        raise ValueError("requested mode count exceeds available degrees of freedom")
    reduced = operator[free][:, free]
    inverse_root = 1 / np.sqrt(mass[free])
    symmetric = diags(inverse_root) @ reduced @ diags(inverse_root)
    initial = np.random.default_rng(7).normal(size=len(free))
    eigenvalues, vectors = eigsh(symmetric, k=count + skip, sigma=-1.0, which="LM", v0=initial, tol=1e-11)
    order = np.argsort(eigenvalues)[skip:]
    eigenvalues, vectors = eigenvalues[order], vectors[:, order] * inverse_root[:, None]
    if np.any(eigenvalues <= 0):
        raise RuntimeError("nonpositive physical cavity eigenvalue")
    fields = np.zeros((count, *grid.shape))
    residuals = []
    for index in range(count):
        field = vectors[:, index]
        if field[np.argmax(abs(field))] < 0:
            field = -field
        fields[index].ravel()[free] = field
        load = eigenvalues[index] * mass[free] * field
        residuals.append(np.linalg.norm(reduced @ field - load) / np.linalg.norm(load))
    return {
        "eigenvalues_m2": eigenvalues,
        "cutoff_hz": C0 * np.sqrt(eigenvalues) / (2 * np.pi),
        "fields": fields,
        "residuals": np.array(residuals),
    }
