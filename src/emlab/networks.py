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


def cascade(*matrices):
    from .core import finite_array

    if not matrices:
        raise ValueError("at least one chain matrix required")
    result = np.eye(2, dtype=complex)
    for matrix in matrices:
        matrix = finite_array(matrix)
        if matrix.shape[-2:] != (2, 2):
            raise ValueError("two by two chain matrices required")
        result = result @ matrix
    return result


def abcd_to_s(matrix, reference=(50.0, 50.0)):
    from .core import finite_array

    matrix = finite_array(matrix)
    z = finite_array(reference, real=True)
    if matrix.shape[-2:] != (2, 2) or z.shape != (2,) or np.any(z <= 0):
        raise ValueError("chain matrices and two positive real port impedances required")
    a, b, c, d = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 1, 0], matrix[..., 1, 1]
    denominator = a * z[1] + b + c * z.prod() + d * z[0]
    if np.any(abs(denominator) == 0):
        raise ValueError("singular scattering conversion")
    factor = 2 * np.sqrt(z.prod())
    result = np.empty(matrix.shape, complex)
    result[..., 0, 0] = (a * z[1] + b - c * z.prod() - d * z[0]) / denominator
    result[..., 1, 1] = (-a * z[1] + b - c * z.prod() + d * z[0]) / denominator
    result[..., 1, 0] = factor / denominator
    result[..., 0, 1] = factor * (a * d - b * c) / denominator
    return result


def s_to_abcd(scattering, reference=(50.0, 50.0)):
    from .core import finite_array

    s = finite_array(scattering)
    z = finite_array(reference, real=True)
    if s.shape[-2:] != (2, 2) or z.shape != (2,) or np.any(z <= 0):
        raise ValueError("scattering matrices and two positive real references required")
    s11, s12, s21, s22 = s[..., 0, 0], s[..., 0, 1], s[..., 1, 0], s[..., 1, 1]
    if np.any(abs(s21) < 1e-14):
        raise ValueError("transmission too small for stable chain conversion; cascade scattering instead")
    result = np.empty(s.shape, complex)
    result[..., 0, 0] = np.sqrt(z[0] / z[1]) * ((1 + s11) * (1 - s22) + s12 * s21) / (2 * s21)
    result[..., 0, 1] = np.sqrt(z.prod()) * ((1 + s11) * (1 + s22) - s12 * s21) / (2 * s21)
    result[..., 1, 0] = ((1 - s11) * (1 - s22) - s12 * s21) / (2 * s21 * np.sqrt(z.prod()))
    result[..., 1, 1] = np.sqrt(z[1] / z[0]) * ((1 - s11) * (1 + s22) + s12 * s21) / (2 * s21)
    return result


def star(left, right):
    from .core import finite_array

    a, b = finite_array(left), finite_array(right)
    if a.shape[-2:] != (2, 2) or b.shape[-2:] != (2, 2):
        raise ValueError("two-port scattering matrices required")
    a, b = np.broadcast_arrays(a, b)
    denominator = 1 - a[..., 1, 1] * b[..., 0, 0]
    if np.any(abs(denominator) < 1e-15):
        raise ValueError("singular internal multiple-reflection resonance")
    result = np.empty(a.shape, complex)
    result[..., 0, 0] = a[..., 0, 0] + a[..., 0, 1] * b[..., 0, 0] * a[..., 1, 0] / denominator
    result[..., 0, 1] = a[..., 0, 1] * b[..., 0, 1] / denominator
    result[..., 1, 0] = b[..., 1, 0] * a[..., 1, 0] / denominator
    result[..., 1, 1] = b[..., 1, 1] + b[..., 1, 0] * a[..., 1, 1] * b[..., 0, 1] / denominator
    return result


def renormalize(scattering, old_reference, new_reference):
    from .core import finite_array

    s = finite_array(scattering)
    old = np.broadcast_to(finite_array(old_reference, real=True), (2,))
    new = np.broadcast_to(finite_array(new_reference, real=True), (2,))
    if s.shape[-2:] != (2, 2) or np.any(old <= 0) or np.any(new <= 0):
        raise ValueError("two-port scattering and positive real reference impedances required")
    ratio = np.sqrt(new / old)
    p = np.diag((ratio + 1 / ratio) / 2)
    q = np.diag((ratio - 1 / ratio) / 2)
    return np.linalg.solve(p - s @ q, s @ p - q)
