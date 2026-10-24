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


def mode_gram(grid, fields, material_weight=1.0):
    from .core import finite_array
    from .grid import dual_weights

    fields = finite_array(fields)
    weight = dual_weights(grid) * np.broadcast_to(finite_array(material_weight, real=True), grid.shape)
    if fields.ndim != 3 or fields.shape[1:] != grid.shape or np.any(weight <= 0):
        raise ValueError("mode fields and positive material quadrature required")
    return np.einsum("iyx,jyx,yx->ij", fields.conj(), fields, weight)


def guide_propagation(frequency_hz, cutoff_hz, epsilon_r=1.0, mu_r=1.0):
    from .core import C0, finite_array

    frequency = finite_array(frequency_hz, real=True)
    if np.any(frequency <= 0) or min(cutoff_hz, epsilon_r, mu_r) <= 0:
        raise ValueError("positive frequency cutoff and material constants required")
    speed = C0 / np.sqrt(epsilon_r * mu_r)
    return 2 * np.pi / speed * np.sqrt((cutoff_hz**2 - frequency**2).astype(complex))


def guide_impedance(frequency_hz, cutoff_hz, polarization="te", epsilon_r=1.0, mu_r=1.0):
    from .core import EPS0, MU0

    gamma = guide_propagation(frequency_hz, cutoff_hz, epsilon_r, mu_r)
    if np.any(abs(gamma) == 0):
        raise ValueError("exact cutoff impedance requires a limiting formulation")
    omega = 2 * np.pi * np.asarray(frequency_hz)
    if polarization == "te":
        return 1j * omega * MU0 * mu_r / gamma
    if polarization == "tm":
        return gamma / (1j * omega * EPS0 * epsilon_r)
    raise ValueError("polarization must be te or tm")


def guide_velocities(frequency_hz, cutoff_hz, epsilon_r=1.0, mu_r=1.0):
    from .core import C0, finite_array

    frequency = finite_array(frequency_hz, real=True)
    guide_propagation(frequency, cutoff_hz, epsilon_r, mu_r)
    if np.any(frequency <= cutoff_hz):
        raise ValueError("propagating frequencies strictly above cutoff required")
    speed = C0 / np.sqrt(epsilon_r * mu_r)
    factor = np.sqrt(1 - (cutoff_hz / frequency) ** 2)
    return speed / factor, speed * factor


def project(grid, field, basis, material_weight=1.0):
    from .core import finite_array
    from .grid import dual_weights

    field, basis = finite_array(field), finite_array(basis)
    if field.shape != grid.shape:
        raise ValueError("field must match the modal grid")
    gram = mode_gram(grid, basis, material_weight)
    if np.linalg.matrix_rank(gram) < len(basis):
        raise ValueError("modal basis is linearly dependent")
    weights = dual_weights(grid) * np.broadcast_to(material_weight, grid.shape)
    load = np.einsum("iyx,yx,yx->i", basis.conj(), field, weights)
    coefficients = np.linalg.solve(gram, load)
    recovered = np.einsum("i,iyx->yx", coefficients, basis)
    return coefficients, recovered


def subspace_overlap(grid, first, second, material_weight=1.0):
    from .core import finite_array
    from .grid import dual_weights

    first, second = finite_array(first), finite_array(second)
    weights = dual_weights(grid) * np.broadcast_to(finite_array(material_weight, real=True), grid.shape)
    if (
        first.ndim != 3
        or second.ndim != 3
        or first.shape[1:] != grid.shape
        or second.shape[1:] != grid.shape
        or np.any(weights <= 0)
    ):
        raise ValueError("two modal groups on the same positively weighted grid required")
    root = np.sqrt(weights.ravel())
    columns_a, columns_b = (
        first.reshape(len(first), -1).T * root[:, None],
        second.reshape(len(second), -1).T * root[:, None],
    )
    if np.linalg.matrix_rank(columns_a) < len(first) or np.linalg.matrix_rank(columns_b) < len(second):
        raise ValueError("independent modes required in each subspace")
    qa, _ = np.linalg.qr(columns_a)
    qb, _ = np.linalg.qr(columns_b)
    return np.clip(np.linalg.svd(qa.conj().T @ qb, compute_uv=False), 0, 1)


def dielectric_shift(grid, electric_field, epsilon_r, delta_epsilon):
    from .core import finite_array
    from .grid import dual_weights

    field = finite_array(electric_field)
    epsilon = np.broadcast_to(finite_array(epsilon_r, real=True), grid.shape)
    change = np.broadcast_to(finite_array(delta_epsilon, real=True), grid.shape)
    if field.shape != grid.shape or np.any(epsilon <= 0):
        raise ValueError("electric eigenfield and positive baseline dielectric map required")
    density = abs(field) ** 2 * dual_weights(grid)
    denominator = float(np.sum(epsilon * density))
    if denominator <= 0:
        raise ValueError("nonzero mode energy required")
    return -0.5 * float(np.sum(change * density)) / denominator


def slab_te_modes(frequency_hz, thickness_m, core_index, cladding_index):
    from scipy.optimize import brentq

    from .core import C0

    if min(frequency_hz, thickness_m, cladding_index) <= 0 or core_index <= cladding_index:
        raise ValueError("positive frequency/thickness and guiding core index contrast required")
    k0 = 2 * np.pi * frequency_hz / C0
    half = thickness_m / 2
    normalized = k0 * half * np.sqrt(core_index**2 - cladding_index**2)
    results = []
    mode = 0
    while mode * np.pi / 2 < normalized * (1 - 1e-14):
        lower, upper = mode * np.pi / 2, min(normalized, (mode + 1) * np.pi / 2)

        def condition(u):
            return u - lower - np.arctan2(np.sqrt(max(normalized**2 - u**2, 0)), u)

        transverse = brentq(condition, lower, upper, xtol=1e-14, rtol=1e-14)
        decay = np.sqrt(max(normalized**2 - transverse**2, 0)) / half
        beta = np.sqrt((k0 * core_index) ** 2 - (transverse / half) ** 2)
        results.append(
            {
                "mode": mode,
                "parity": "even" if mode % 2 == 0 else "odd",
                "effective_index": float(beta / k0),
                "beta_m": float(beta),
                "decay_m": float(decay),
                "transverse_m": float(transverse / half),
            }
        )
        mode += 1
    return results
