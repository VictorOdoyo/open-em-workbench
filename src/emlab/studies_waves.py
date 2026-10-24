import numpy as np


def coating(scale=1, seed=42):
    from .layers import antireflection, stack
    from .results import curve, make

    center = 10e9
    film = antireflection(center, 1, 2)
    frequency = np.linspace(0.5 * center, 1.5 * center, 200 * scale + 1)
    coated = stack(frequency, [film["epsilon_r"]], [film["thickness_m"]], epsilon_substrate=4)
    bare = stack(frequency, [], [], epsilon_substrate=4)
    center_result = stack(center, [film["epsilon_r"]], [film["thickness_m"]], epsilon_substrate=4)
    rows = [
        {
            "frequency_hz": f,
            "bare_reflectance": r0,
            "coated_reflectance": r,
            "transmittance": t,
            "absorption": a,
        }
        for f, r0, r, t, a in zip(frequency, bare["R"], coated["R"], coated["T"], coated["A"], strict=True)
    ]
    return make(
        "multilayer-coating",
        {
            "center_hz": center,
            "epsilon_incident": 1,
            "epsilon_substrate": 4,
            "film_epsilon_r": film["epsilon_r"],
            "film_thickness_m": film["thickness_m"],
            "incident_peak_e_v_m": 1,
            "seed": seed,
        },
        {
            "design_reflectance": center_result["R"],
            "maximum_power_balance_defect": np.max(abs(coated["R"] + coated["T"] - 1)),
        },
        {
            "frequency_ghz": frequency / 1e9,
            "bare_r": bare["R"],
            "coated_r": coated["R"],
            "transmittance": coated["T"],
            "absorption": coated["A"],
            "s": coated["s"],
        },
        [
            curve(
                "Antireflection matching is frequency selective",
                "frequency_ghz",
                [("bare_r", "bare interface"), ("coated_r", "quarter-wave film")],
                "Frequency (GHz)",
                "Power reflectance",
            ),
            curve(
                "Lossless coating conserves incident power",
                "frequency_ghz",
                [("coated_r", "R"), ("transmittance", "T"), ("absorption", "1-R-T")],
                "Frequency (GHz)",
                "Normalized power",
            ),
        ],
        {"spectrum": rows},
        [
            "Normal incidence, isotropic lossless dielectrics, plane waves and infinite transverse extent.",
            "Tangential electric amplitudes are converted to transmitted power using exterior admittance.",
        ],
    )


def bragg(scale=1, seed=42):
    from .layers import bloch_wavenumber, bragg_mirror, fields, stack
    from .results import curve, make

    center, pairs = 10e9, 6
    epsilon, thickness = bragg_mirror(center, 3, 1.5, pairs)
    frequency = np.linspace(0.4 * center, 1.6 * center, 240 * scale + 1)
    result = stack(frequency, epsilon, thickness)
    bloch = bloch_wavenumber(frequency, epsilon[:2], thickness[:2])
    z, electric = [], []
    position = 0.0
    for index, distance in enumerate(thickness):
        coordinate = np.linspace(0, distance, 40)
        field = fields(center, epsilon, thickness, index, coordinate)
        z.extend(position + coordinate)
        electric.extend(abs(field["electric"]))
        position += distance
    design = stack(center, epsilon, thickness)
    rows = [
        {"frequency_hz": f, "reflectance": r, "transmittance": t, "bloch_attenuation_m": attenuation}
        for f, r, t, attenuation in zip(frequency, result["R"], result["T"], -bloch.imag, strict=True)
    ]
    return make(
        "bragg-mirror",
        {
            "center_hz": center,
            "index_high": 3,
            "index_low": 1.5,
            "pairs": pairs,
            "epsilon_layers": epsilon,
            "thickness_m": thickness,
            "incident_peak_e_v_m": 1,
            "seed": seed,
        },
        {
            "design_reflectance": design["R"],
            "design_transmittance": design["T"],
            "maximum_lossless_balance_defect": np.max(abs(result["A"])),
        },
        {
            "frequency_ghz": frequency / 1e9,
            "r": result["R"],
            "t": result["T"],
            "bloch_attenuation": -bloch.imag,
            "position_mm": np.array(z) * 1e3,
            "electric_magnitude": electric,
            "s": result["s"],
        },
        [
            curve(
                "Six-pair reflector stopband",
                "frequency_ghz",
                [("r", "R"), ("t", "T")],
                "Frequency (GHz)",
                "Normalized power",
            ),
            curve(
                "Infinite-periodic-cell evanescence",
                "frequency_ghz",
                [("bloch_attenuation", "Bloch attenuation")],
                "Frequency (GHz)",
                "Attenuation (1/m)",
            ),
            curve(
                "Finite stack field at design frequency",
                "position_mm",
                [("electric_magnitude", "|E|")],
                "Distance through stack (mm)",
                "Electric field (V/m)",
            ),
        ],
        {"spectrum": rows},
        [
            "The Bloch calculation describes an infinite periodic unit cell; R/T describe the finite stack.",
            "Dielectrics are nondispersive and lossless in this demonstration.",
        ],
    )
