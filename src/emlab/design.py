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
