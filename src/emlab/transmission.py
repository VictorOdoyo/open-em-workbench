import numpy as np


def rlgc(frequency_hz, resistance, inductance, conductance, capacitance):
    from .core import finite_array

    frequency = finite_array(frequency_hz, real=True)
    r, inductance_value, g, c = [
        finite_array(v, real=True) for v in (resistance, inductance, conductance, capacitance)
    ]
    if (
        np.any(frequency <= 0)
        or np.any(r < 0)
        or np.any(g < 0)
        or np.any(inductance_value <= 0)
        or np.any(c <= 0)
    ):
        raise ValueError("positive frequency/L/C and nonnegative R/G required")
    series = r + 2j * np.pi * frequency * inductance_value
    shunt = g + 2j * np.pi * frequency * c
    gamma = np.sqrt(series * shunt)
    impedance = np.sqrt(series / shunt)
    return gamma, impedance


def line_matrix(gamma, characteristic_impedance, length_m):
    from .core import finite_array

    gamma, impedance = finite_array(gamma), finite_array(characteristic_impedance)
    if length_m < 0 or not np.isfinite(length_m) or np.any(impedance.real <= 0) or np.any(gamma.real < 0):
        raise ValueError("passive line parameters and nonnegative length required")
    gamma, impedance = np.broadcast_arrays(gamma, impedance)
    if np.any((gamma * length_m).real > 300):
        raise ValueError("line too attenuating for chain matrices; use line_scattering")
    phase = gamma * length_m
    result = np.empty((*phase.shape, 2, 2), complex)
    result[..., 0, 0] = result[..., 1, 1] = np.cosh(phase)
    result[..., 0, 1] = impedance * np.sinh(phase)
    result[..., 1, 0] = np.sinh(phase) / impedance
    return result


def line_scattering(gamma, characteristic_impedance, length_m, reference=50.0):
    from .core import finite_array

    gamma, impedance = finite_array(gamma), finite_array(characteristic_impedance)
    if (
        length_m < 0
        or not np.isfinite(length_m)
        or not np.isfinite(reference)
        or reference <= 0
        or np.any(gamma.real < 0)
        or np.any(impedance.real <= 0)
    ):
        raise ValueError("passive line and positive real port reference required")
    gamma, impedance = np.broadcast_arrays(gamma, impedance)
    wave = np.exp(-gamma * length_m)
    mismatch = (impedance - reference) / (impedance + reference)
    denominator = 1 - mismatch**2 * wave**2
    if np.any(abs(denominator) < 1e-15):
        raise ValueError("singular line multiple-reflection denominator")
    reflection = mismatch * (1 - wave**2) / denominator
    transmission = (1 - mismatch**2) * wave / denominator
    result = np.empty((*wave.shape, 2, 2), complex)
    result[..., 0, 0] = result[..., 1, 1] = reflection
    result[..., 1, 0] = result[..., 0, 1] = transmission
    return result


def input_impedance(gamma, characteristic_impedance, length_m, load_impedance):
    from .core import finite_array

    gamma, impedance = finite_array(gamma), finite_array(characteristic_impedance)
    load = np.asarray(load_impedance, dtype=complex)
    if length_m < 0 or np.any(~np.isfinite(load) & ~np.isposinf(load.real)):
        raise ValueError("nonnegative length and finite load or positive-real open required")
    opened = np.isposinf(load.real)
    safe = np.where(opened, 0, load)
    tangent = np.tanh(gamma * length_m)
    with np.errstate(divide="ignore", invalid="ignore"):
        regular = impedance * (safe + impedance * tangent) / (impedance + safe * tangent)
        open_value = impedance / tangent
    return np.where(opened, open_value, regular)


def reflection_along(gamma, length_m, positions_m, load_reflection):
    from .core import finite_array

    gamma = finite_array(gamma)
    positions = finite_array(positions_m, real=True)
    reflected = finite_array(load_reflection)
    if length_m < 0 or np.any((positions < 0) | (positions > length_m)) or np.any(gamma.real < 0):
        raise ValueError("passive line and positions within its physical extent required")
    return reflected * np.exp(-2 * gamma * (length_m - positions))


def coax_parameters(inner_radius_m, outer_radius_m, epsilon_r=1, mu_r=1):
    from .capacitance import coaxial
    from .core import MU0

    capacitance = coaxial(inner_radius_m, outer_radius_m, epsilon_r)
    if mu_r <= 0:
        raise ValueError("positive permeability required")
    inductance = MU0 * mu_r / (2 * np.pi) * np.log(outer_radius_m / inner_radius_m)
    return {
        "inductance_h_m": inductance,
        "capacitance_f_m": capacitance,
        "impedance_ohm": np.sqrt(inductance / capacitance),
        "velocity_m_s": 1 / np.sqrt(inductance * capacitance),
    }


def coax_losses(
    frequency_hz, inner_radius_m, outer_radius_m, epsilon_r=1, conductivity=5.8e7, loss_tangent=0, mu_r=1
):
    from .materials import surface_impedance

    parameters = coax_parameters(inner_radius_m, outer_radius_m, epsilon_r, mu_r)
    if loss_tangent < 0:
        raise ValueError("nonnegative dielectric loss tangent required")
    surface = surface_impedance(frequency_hz, conductivity)
    resistance = surface.real / (2 * np.pi) * (1 / inner_radius_m + 1 / outer_radius_m)
    conductance = 2 * np.pi * np.asarray(frequency_hz) * parameters["capacitance_f_m"] * loss_tangent
    return {"resistance_ohm_m": resistance, "conductance_s_m": conductance, **parameters}


def quarter_wave(source_impedance, load_impedance, center_hz, velocity_m_s):
    if min(source_impedance, load_impedance, center_hz, velocity_m_s) <= 0:
        raise ValueError("positive resistive matching impedances frequency and velocity required")
    return {
        "impedance_ohm": np.sqrt(source_impedance * load_impedance),
        "length_m": velocity_m_s / (4 * center_hz),
    }


def tapered_line(
    frequency_hz, start_impedance, end_impedance, length_m, velocity_m_s, segments=32, reference=50.0
):
    from .core import finite_array
    from .networks import star

    f = finite_array(frequency_hz, real=True)
    if (
        np.any(f <= 0)
        or min(start_impedance, end_impedance, length_m, velocity_m_s) <= 0
        or not isinstance(segments, int)
        or segments < 1
    ):
        raise ValueError("positive taper parameters and section count required")
    locations = (np.arange(segments) + 0.5) / segments
    impedances = start_impedance * (end_impedance / start_impedance) ** locations
    result = np.broadcast_to(np.array([[0, 1], [1, 0]], complex), (*f.shape, 2, 2)).copy()
    for impedance in impedances:
        result = star(
            result, line_scattering(2j * np.pi * f / velocity_m_s, impedance, length_m / segments, reference)
        )
    return result
