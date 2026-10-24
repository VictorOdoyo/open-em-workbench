import numpy as np


def series(impedance):
    from .core import finite_array

    impedance = finite_array(impedance)
    result = np.broadcast_to(np.eye(2, dtype=complex), (*impedance.shape, 2, 2)).copy()
    result[..., 0, 1] = impedance
    return result


def shunt(admittance):
    from .core import finite_array

    admittance = finite_array(admittance)
    result = np.broadcast_to(np.eye(2, dtype=complex), (*admittance.shape, 2, 2)).copy()
    result[..., 1, 0] = admittance
    return result
