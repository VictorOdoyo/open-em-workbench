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
