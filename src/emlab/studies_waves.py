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
