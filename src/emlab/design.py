import numpy as np


def bounded_mean(values, lower, upper, target_mean, weights=None):
    from .core import finite_array

    values = finite_array(values, real=True)
    weight = (
        np.ones(values.shape)
        if weights is None
        else np.broadcast_to(finite_array(weights, real=True), values.shape)
    )
    if (
        not np.all(np.isfinite([lower, upper, target_mean]))
        or not lower < upper
        or not lower <= target_mean <= upper
        or np.any(weight <= 0)
        or values.size == 0
    ):
        raise ValueError("ordered finite bounds feasible mean and positive weights required")
    low, high = lower - float(values.max()), upper - float(values.min())
    for _ in range(80):
        shift = (low + high) / 2
        candidate = np.clip(values + shift, lower, upper)
        mean = float(np.sum(candidate * weight) / weight.sum())
        if mean < target_mean:
            low = shift
        else:
            high = shift
    return np.clip(values + (low + high) / 2, lower, upper)


def optimize_dielectric(
    grid, fixed, voltage, design_mask, initial_epsilon=2.5, bounds=(1.0, 4.0), iterations=12, initial_step=0.5
):
    from .core import finite_array
    from .electrostatics import energy, energy_sensitivity, solve
    from .grid import dual_weights

    fixed, mask = np.asarray(fixed), np.asarray(design_mask)
    if (
        fixed.shape != grid.shape
        or fixed.dtype != bool
        or mask.shape != grid.shape
        or mask.dtype != bool
        or not np.any(mask)
        or np.any(mask & fixed)
    ):
        raise ValueError("nonempty free-node design mask and boolean fixed-potential mask required")
    if not isinstance(iterations, int) or iterations < 1 or initial_step <= 0:
        raise ValueError("positive optimization iteration count and step required")
    epsilon = np.broadcast_to(finite_array(initial_epsilon, real=True), grid.shape).copy()
    lower, upper = bounds
    if not np.all(np.isfinite([lower, upper])) or not 0 < lower < upper:
        raise ValueError("positive ordered finite dielectric bounds required")
    if np.any((epsilon[mask] < lower) | (epsilon[mask] > upper)) or np.any(epsilon <= 0):
        raise ValueError("initial material must satisfy positive bounds")
    weight = dual_weights(grid)[mask]
    target = float(np.sum(epsilon[mask] * weight) / weight.sum())
    potential = solve(grid, epsilon, fixed, voltage)["potential"]
    objective = energy(grid, potential, epsilon)
    history = [{"iteration": 0, "energy_j_m": objective, "step": 0.0}]
    for iteration in range(1, iterations + 1):
        gradient = energy_sensitivity(grid, potential, epsilon)
        scale = float(np.max(abs(gradient[mask])))
        if scale == 0:
            break
        step = initial_step
        accepted = False
        for _ in range(14):
            candidate = epsilon.copy()
            candidate[mask] = bounded_mean(
                epsilon[mask] + step * gradient[mask] / scale, lower, upper, target, weight
            )
            predicted = float(np.sum(gradient * (candidate - epsilon)))
            if predicted <= 0:
                break
            field = solve(grid, candidate, fixed, voltage)["potential"]
            trial = energy(grid, field, candidate)
            if trial >= objective + 1e-4 * predicted:
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break
        epsilon, potential, objective = candidate, field, trial
        history.append({"iteration": iteration, "energy_j_m": objective, "step": step})
    return {"epsilon_r": epsilon, "potential": potential, "history": history, "material_mean": target}
