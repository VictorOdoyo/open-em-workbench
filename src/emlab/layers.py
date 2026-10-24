import numpy as np


def interface(admittance_left, admittance_right):
    from .core import finite_array

    left, right = np.broadcast_arrays(finite_array(admittance_left), finite_array(admittance_right))
    denominator = left + right
    if np.any(abs(denominator) < np.finfo(float).tiny):
        raise ValueError("singular interface admittance")
    result = np.empty((*left.shape, 2, 2), complex)
    result[..., 0, 0] = (left - right) / denominator
    result[..., 1, 1] = -result[..., 0, 0]
    result[..., 1, 0] = 2 * left / denominator
    result[..., 0, 1] = 2 * right / denominator
    return result


def propagation(kz, thickness_m):
    from .core import finite_array

    kz = finite_array(kz).astype(complex)
    if thickness_m < 0 or not np.isfinite(thickness_m) or np.any(kz.imag > 1e-12):
        raise ValueError("nonnegative thickness and outgoing passive longitudinal wavenumber required")
    wave = np.exp(-1j * kz * thickness_m)
    result = np.zeros((*wave.shape, 2, 2), complex)
    result[..., 0, 1] = result[..., 1, 0] = wave
    return result
