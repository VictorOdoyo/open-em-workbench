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


def absorbing_stack(scale=1, seed=42):
    from .core import EPS0
    from .layers import absorption_by_layer, fields, stack
    from .results import curve, make

    center = 10e9
    epsilon, thickness = [2 - 0.3j, 4 - 0.8j], [0.01, 0.008]
    frequency = np.linspace(0.5 * center, 1.5 * center, 200 * scale + 1)
    spectrum = stack(frequency, epsilon, thickness)
    design = stack(center, epsilon, thickness)
    absorbed = absorption_by_layer(center, epsilon, thickness)
    profile_z, profile_e, profile_flux, rows = [], [], [], []
    position = 0.0
    for index, distance in enumerate(thickness):
        local = np.linspace(0, distance, 400 * scale + 1)
        field = fields(center, epsilon, thickness, index, local)
        joule = (
            np.trapezoid(
                0.5 * 2 * np.pi * center * EPS0 * (-epsilon[index].imag) * abs(field["electric"]) ** 2, local
            )
            / field["incident_flux_w_m2"]
        )
        rows.append({"layer": index, "flux_absorption": absorbed[index], "joule_absorption": joule})
        profile_z.extend(position + local)
        profile_e.extend(abs(field["electric"]))
        profile_flux.extend(field["flux_w_m2"] / field["incident_flux_w_m2"])
        position += distance
    defect = max(abs(row["flux_absorption"] - row["joule_absorption"]) for row in rows)
    return make(
        "absorbing-stack",
        {
            "center_hz": center,
            "epsilon_layers_real_imag": [[e.real, e.imag] for e in epsilon],
            "thickness_m": thickness,
            "incident_peak_e_v_m": 1,
            "seed": seed,
        },
        {
            "total_absorption": design["A"],
            "layer_absorption_sum": np.sum(absorbed),
            "maximum_joule_flux_defect": defect,
        },
        {
            "frequency_ghz": frequency / 1e9,
            "r": spectrum["R"],
            "t": spectrum["T"],
            "a": spectrum["A"],
            "position_mm": np.array(profile_z) * 1e3,
            "electric": profile_e,
            "flux": profile_flux,
        },
        [
            curve(
                "Passive lossy stack",
                "frequency_ghz",
                [("r", "R"), ("t", "T"), ("a", "A")],
                "Frequency (GHz)",
                "Normalized power",
            ),
            curve(
                "Internal electric standing wave",
                "position_mm",
                [("electric", "|E|")],
                "Distance through stack (mm)",
                "Electric field (V/m)",
            ),
            curve(
                "Local energy flow decreases through loss",
                "position_mm",
                [("flux", "normalized flux")],
                "Distance through stack (mm)",
                "Poynting flux / incident flux",
            ),
        ],
        {
            "layers": rows,
            "spectrum": [
                {"frequency_hz": f, "R": r, "T": t, "A": a}
                for f, r, t, a in zip(frequency, spectrum["R"], spectrum["T"], spectrum["A"], strict=True)
            ],
        },
        [
            "Complex permittivity is a finite-band phenomenological model, not a causal broadband fit.",
            "Independent Joule integration uses peak phasors and the factor one half.",
        ],
    )


def microwave_lines(scale=1, seed=42):
    from .core import C0, to_db
    from .networks import abcd_to_s, group_delay, scattering_audit
    from .results import curve, make
    from .transmission import coax_losses, line_matrix, line_scattering, quarter_wave, rlgc

    frequency = np.linspace(0.1e9, 3e9, 400 * scale + 1)
    parameters = coax_losses(frequency, 0.001, 0.003, 2.25, loss_tangent=0.002)
    gamma, impedance = rlgc(
        frequency,
        parameters["resistance_ohm_m"],
        parameters["inductance_h_m"],
        parameters["conductance_s_m"],
        parameters["capacitance_f_m"],
    )
    scattering = line_scattering(gamma, impedance, 1.0, reference=50.0)
    delay = group_delay(frequency, scattering[:, 1, 0])
    matching = quarter_wave(50, 100, 1e9, C0)
    transformer = abcd_to_s(
        line_matrix(2j * np.pi * frequency / C0, matching["impedance_ohm"], matching["length_m"]), (50, 100)
    )
    at_design = abcd_to_s(
        line_matrix(2j * np.pi * 1e9 / C0, matching["impedance_ohm"], matching["length_m"]), (50, 100)
    )
    rows = [
        {
            "frequency_hz": f,
            "s21_db": db,
            "group_delay_s": d,
            "resistance_ohm_m": r,
            "matching_reflectance": refl,
        }
        for f, db, d, r, refl in zip(
            frequency,
            to_db(scattering[:, 1, 0], field=True),
            delay,
            parameters["resistance_ohm_m"],
            abs(transformer[:, 0, 0]) ** 2,
            strict=True,
        )
    ]
    return make(
        "microwave-lines",
        {
            "coax_inner_m": 0.001,
            "coax_outer_m": 0.003,
            "epsilon_r": 2.25,
            "length_m": 1.0,
            "conductivity_s_m": 5.8e7,
            "loss_tangent": 0.002,
            "port_reference_ohm": 50,
            "matching_center_hz": 1e9,
            "matching_length_m": matching["length_m"],
            "seed": seed,
        },
        {
            "minimum_passivity_margin": np.min(scattering_audit(scattering)["passivity_margin"]),
            "design_matching_reflectance": abs(at_design[0, 0]) ** 2,
            "median_group_delay_s": np.median(delay),
        },
        {
            "frequency_ghz": frequency / 1e9,
            "s21_db": to_db(scattering[:, 1, 0], field=True),
            "delay_ns": delay * 1e9,
            "resistance": parameters["resistance_ohm_m"],
            "matching_r": abs(transformer[:, 0, 0]) ** 2,
            "s": scattering,
        },
        [
            curve(
                "One-metre coaxial transmission",
                "frequency_ghz",
                [("s21_db", "|S21|")],
                "Frequency (GHz)",
                "Transmission (dB)",
            ),
            curve(
                "Transmission group delay",
                "frequency_ghz",
                [("delay_ns", "group delay")],
                "Frequency (GHz)",
                "Group delay (ns)",
            ),
            curve(
                "Skin-effect series resistance",
                "frequency_ghz",
                [("resistance", "copper resistance")],
                "Frequency (GHz)",
                "Resistance (ohm/m)",
            ),
            curve(
                "Separate 50-to-100 ohm quarter-wave transformer",
                "frequency_ghz",
                [("matching_r", "|S11|^2")],
                "Frequency (GHz)",
                "Power reflectance",
            ),
        ],
        {"spectrum": rows},
        [
            "Coax loss uses the good-conductor skin approximation and frequency-specific dielectric conductance.",
            "The matching transformer is a separate ideal line, not the same coaxial geometry.",
        ],
    )
