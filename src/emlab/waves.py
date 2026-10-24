import numpy as np


def fresnel_normal(impedance_left, impedance_right):
    from .core import finite_array

    left, right = finite_array(impedance_left).astype(complex), finite_array(impedance_right).astype(complex)
    if np.any(left.real <= 0) or np.any(right.real <= 0) or np.any(abs(left + right) < 1e-30):
        raise ValueError("positive-real propagating wave impedances required")
    reflection = (right - left) / (right + left)
    transmission = 1 + reflection
    reflectance = abs(reflection) ** 2
    transmittance = (1 / right).real / (1 / left).real * abs(transmission) ** 2
    return {"r": reflection, "t": transmission, "R": reflectance, "T": transmittance}
