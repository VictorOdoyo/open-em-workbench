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


def components(
    frequency_hz,
    epsilon_layers,
    thickness_m,
    epsilon_incident=1.0,
    epsilon_substrate=1.0,
    angle_radians=0.0,
    polarization="te",
):
    from .core import C0, EPS0, MU0, finite_array
    from .waves import longitudinal_wavenumber

    frequency = finite_array(frequency_hz, real=True)
    thickness = finite_array(thickness_m, real=True, ndim=1)
    if (
        frequency.ndim > 1
        or np.any(frequency <= 0)
        or len(epsilon_layers) != len(thickness)
        or np.any(thickness < 0)
        or min(epsilon_incident, epsilon_substrate) <= 0
        or not 0 <= angle_radians < np.pi / 2
    ):
        raise ValueError("aligned layers and passive exterior half spaces required")
    epsilon = [
        np.broadcast_to(finite_array(value), frequency.shape)
        for value in [epsilon_incident, *epsilon_layers, epsilon_substrate]
    ]
    omega = 2 * np.pi * frequency
    transverse = omega / C0 * np.sqrt(epsilon_incident) * np.sin(angle_radians)
    kz = [longitudinal_wavenumber(frequency, value, transverse_wavenumber=transverse) for value in epsilon]
    if any(np.any(abs(value) < 1e-14) for value in kz):
        raise ValueError("exact grazing/cutoff admittance requires a limiting formulation")
    if polarization == "te":
        admittance = [value / (omega * MU0) for value in kz]
    elif polarization == "tm":
        admittance = [omega * EPS0 * eps / value for eps, value in zip(epsilon, kz, strict=True)]
    else:
        raise ValueError("polarization must be te or tm")
    pieces = []
    for index in range(len(thickness)):
        pieces.extend(
            [
                interface(admittance[index], admittance[index + 1]),
                propagation(kz[index + 1], float(thickness[index])),
            ]
        )
    pieces.append(interface(admittance[-2], admittance[-1]))
    return {"pieces": pieces, "admittance": admittance, "kz": kz, "thickness": thickness}


def _compose(pieces):
    from .networks import star

    result = np.array([[0, 1], [1, 0]], complex)
    for piece in pieces:
        result = star(result, piece)
    return result


def stack(frequency_hz, epsilon_layers, thickness_m, **options):
    system = components(frequency_hz, epsilon_layers, thickness_m, **options)
    raw = _compose(system["pieces"])
    left, right = system["admittance"][0].real, system["admittance"][-1].real
    if np.any(left <= 0) or np.any(right <= 0):
        raise ValueError("propagating exterior ports required for transmitted power")
    normalization = np.sqrt(np.stack((left, right), axis=-1))
    scattering = raw * normalization[..., :, None] / normalization[..., None, :]
    r, t = raw[..., 0, 0], raw[..., 1, 0]
    reflectance, transmittance = abs(r) ** 2, right / left * abs(t) ** 2
    return {
        "r": r,
        "t": t,
        "R": reflectance,
        "T": transmittance,
        "A": 1 - reflectance - transmittance,
        "s": scattering,
    }


def fields(frequency_hz, epsilon_layers, thickness_m, layer_index, positions_m, **options):
    from .core import finite_array

    if (
        np.ndim(frequency_hz) != 0
        or not isinstance(layer_index, int)
        or not 0 <= layer_index < len(epsilon_layers)
    ):
        raise ValueError("one frequency and a valid layer index required")
    system = components(frequency_hz, epsilon_layers, thickness_m, **options)
    distance = finite_array(positions_m, real=True)
    thickness = float(system["thickness"][layer_index])
    if np.any((distance < 0) | (distance > thickness)):
        raise ValueError("field coordinates must lie inside the selected layer")
    prefix = _compose(system["pieces"][: 2 * layer_index + 1])
    suffix = _compose(system["pieces"][2 * layer_index + 2 :])
    kz, admittance = system["kz"][layer_index + 1], system["admittance"][layer_index + 1]
    wave = np.exp(-1j * kz * thickness)
    forward_left = prefix[1, 0] / (1 - prefix[1, 1] * wave**2 * suffix[0, 0])
    backward_right = suffix[0, 0] * forward_left * wave
    forward = forward_left * np.exp(-1j * kz * distance)
    backward = backward_right * np.exp(-1j * kz * (thickness - distance))
    electric = forward + backward
    magnetic = admittance * (forward - backward)
    return {
        "electric": electric,
        "magnetic": magnetic,
        "flux_w_m2": 0.5 * np.real(electric * magnetic.conj()),
        "incident_flux_w_m2": float(0.5 * system["admittance"][0].real),
    }


def absorption_by_layer(frequency_hz, epsilon_layers, thickness_m, **options):
    absorbed = []
    for index, thickness in enumerate(thickness_m):
        field = fields(frequency_hz, epsilon_layers, thickness_m, index, [0, thickness], **options)
        absorbed.append(float((field["flux_w_m2"][0] - field["flux_w_m2"][1]) / field["incident_flux_w_m2"]))
    return np.array(absorbed)


def antireflection(center_hz, index_incident, index_substrate):
    from .core import C0

    if min(center_hz, index_incident, index_substrate) <= 0:
        raise ValueError("positive design frequency and refractive indices required")
    index = np.sqrt(index_incident * index_substrate)
    return {"epsilon_r": float(index**2), "thickness_m": float(C0 / (4 * center_hz * index))}


def bragg_mirror(center_hz, index_high, index_low, pairs=6):
    from .core import C0

    if (
        min(center_hz, index_low) <= 0
        or index_high <= index_low
        or not isinstance(pairs, int)
        or isinstance(pairs, bool)
        or pairs < 1
    ):
        raise ValueError("positive ordered indices and Bragg pair count required")
    epsilon = [index_high**2, index_low**2] * pairs
    thickness = [C0 / (4 * center_hz * index_high), C0 / (4 * center_hz * index_low)] * pairs
    return epsilon, thickness
