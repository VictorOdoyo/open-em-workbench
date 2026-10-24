import numpy as np

C0 = 299792458.0
MU0 = 1.25663706127e-6
EPS0 = 1 / (MU0 * C0**2)
ETA0 = MU0 * C0


def wavelength(frequency_hz, refractive_index=1):
    frequency = np.asarray(frequency_hz, dtype=float)
    if np.any(~np.isfinite(frequency)) or np.any(frequency <= 0) or refractive_index <= 0:
        raise ValueError("positive finite frequency and refractive index required")
    return C0 / (frequency * refractive_index)


def frequency(wavelength_m, refractive_index=1):
    length = np.asarray(wavelength_m, dtype=float)
    if np.any(~np.isfinite(length)) or np.any(length <= 0) or refractive_index <= 0:
        raise ValueError("positive finite wavelength and refractive index required")
    return C0 / (length * refractive_index)


def finite_array(values, *, real=False, ndim=None):
    array = np.asarray(values)
    if array.dtype.kind not in "biufc" or not np.all(np.isfinite(array)):
        raise ValueError("finite numeric array required")
    if real and np.iscomplexobj(array) and np.any(array.imag != 0):
        raise ValueError("real values required")
    if ndim is not None and array.ndim != ndim:
        raise ValueError("unexpected array rank")
    return array.real.astype(float) if real else array


def to_db(values, *, field=False):
    values = finite_array(values)
    magnitude = abs(values) if field else values
    if np.iscomplexobj(magnitude) or np.any(magnitude < 0):
        raise ValueError("nonnegative real power ratios required")
    with np.errstate(divide="ignore"):
        return (20 if field else 10) * np.log10(magnitude)


def from_db(values, *, field=False):
    return 10.0 ** (finite_array(values, real=True) / (20 if field else 10))


def relative_error(reference, observed, weights=None):
    reference, observed = finite_array(reference), finite_array(observed)
    if reference.shape != observed.shape:
        raise ValueError("matching field shapes required")
    weights = (
        np.ones(reference.shape)
        if weights is None
        else np.broadcast_to(finite_array(weights, real=True), reference.shape)
    )
    if np.any(weights < 0):
        raise ValueError("nonnegative quadrature weights required")
    denominator = np.sum(weights * abs(reference) ** 2)
    if denominator <= 0:
        raise ValueError("nonzero reference energy required")
    return float(np.sqrt(np.sum(weights * abs(observed - reference) ** 2) / denominator))
