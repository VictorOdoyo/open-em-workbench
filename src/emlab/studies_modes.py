import numpy as np


def cavity_cutoffs(scale=1, seed=42):
    from .grid import from_bounds
    from .modes import cavity_modes, rectangular_cutoff
    from .results import curve, image, make

    grid = from_bounds((0, 0.02), (0, 0.012), 64 * scale + 1, 48 * scale + 1)
    arrays, rows, solutions = {"mode_index": np.arange(1, 5)}, [], {}
    for family in ("te", "tm"):
        solved = cavity_modes(grid, polarization=family, count=4)
        solutions[family] = solved
        analytic = []
        for mx in range(5):
            for my in range(5):
                if (family == "te" and mx + my > 0) or (family == "tm" and min(mx, my) > 0):
                    analytic.append((rectangular_cutoff(0.02, 0.012, mx, my, family), mx, my))
        analytic.sort()
        reference = np.array([row[0] for row in analytic[:4]])
        arrays[family + "_numerical"] = solved["cutoff_hz"] / 1e9
        arrays[family + "_reference"] = reference / 1e9
        arrays[family + "_map"] = solved["fields"][0] / np.max(abs(solved["fields"][0]))
        for index, (exact, mx, my) in enumerate(analytic[:4]):
            rows.append(
                {
                    "family": family,
                    "mode_x": mx,
                    "mode_y": my,
                    "reference_hz": exact,
                    "numerical_hz": solved["cutoff_hz"][index],
                    "relative_error": abs(solved["cutoff_hz"][index] / exact - 1),
                }
            )
    return make(
        "cavity-cutoffs",
        {
            "width_m": 0.02,
            "height_m": 0.012,
            "epsilon_r": 1,
            "mu_r": 1,
            "grid_nodes": [grid.nx, grid.ny],
            "modes_per_family": 4,
            "seed": seed,
        },
        {
            "maximum_relative_cutoff_error": max(row["relative_error"] for row in rows),
            "maximum_eigen_residual": max(np.max(value["residuals"]) for value in solutions.values()),
        },
        arrays,
        [
            curve(
                "TE cutoffs with Neumann magnetic boundary",
                "mode_index",
                [("te_numerical", "finite volume"), ("te_reference", "analytic reference")],
                "Mode order",
                "Cutoff frequency (GHz)",
            ),
            curve(
                "TM cutoffs with Dirichlet electric boundary",
                "mode_index",
                [("tm_numerical", "finite volume"), ("tm_reference", "analytic reference")],
                "Mode order",
                "Cutoff frequency (GHz)",
            ),
            image("Fundamental TE magnetic eigenfield", "te_map", grid, "Normalized Hz"),
            image("Fundamental TM electric eigenfield", "tm_map", grid, "Normalized Ez"),
        ],
        {"cutoffs": rows},
        [
            "The constant Neumann solution is removed; TE00 is not a physical hollow-guide TEM mode.",
            "Scalar cutoff/cavity equations do not describe general hybrid inhomogeneous-guide propagation.",
        ],
    )


def slab_guidance(scale=1, seed=42):
    from .modes import slab_fd, slab_te_modes
    from .results import curve, make

    frequency, thickness = 1e14, 2e-6
    exact = slab_te_modes(frequency, thickness, 1.5, 1.45)[0]["effective_index"]
    rows = []
    for base in (321, 641, 1281):
        count = (base - 1) * scale + 1
        x = np.linspace(-16e-6, 16e-6, count)
        epsilon = np.where(abs(x) < thickness / 2, 1.5**2, 1.45**2)
        epsilon[np.isclose(abs(x), thickness / 2, rtol=0, atol=1e-15)] = (1.5**2 + 1.45**2) / 2
        solved = slab_fd(frequency, x, epsilon, 1.45)
        measured = solved["effective_index"][0]
        rows.append(
            {
                "nodes": count,
                "spacing_m": x[1] - x[0],
                "effective_index": measured,
                "relative_error": abs(measured / exact - 1),
                "guided_modes_found": len(solved["effective_index"]),
            }
        )
    mode = solved["fields"][0] / np.max(abs(solved["fields"][0]))
    return make(
        "slab-guidance",
        {
            "frequency_hz": frequency,
            "core_thickness_m": thickness,
            "core_index": 1.5,
            "cladding_index": 1.45,
            "domain_bounds_m": [-16e-6, 16e-6],
            "mesh_nodes": [row["nodes"] for row in rows],
            "seed": seed,
        },
        {
            "analytic_effective_index": exact,
            "finest_effective_index": measured,
            "finest_relative_error": rows[-1]["relative_error"],
        },
        {
            "spacing_nm": [r["spacing_m"] * 1e9 for r in rows],
            "effective_index": [r["effective_index"] for r in rows],
            "reference_index": np.full(len(rows), exact),
            "error": [r["relative_error"] for r in rows],
            "x_um": x * 1e6,
            "mode": mode,
            "refractive_index": np.sqrt(epsilon),
        },
        [
            curve(
                "TE slab effective-index refinement",
                "spacing_nm",
                [("effective_index", "finite difference"), ("reference_index", "transcendental reference")],
                "Grid spacing (nm)",
                "Effective index",
            ),
            curve(
                "Discretization and finite-cladding error",
                "spacing_nm",
                [("error", "relative error")],
                "Grid spacing (nm)",
                "Effective-index relative error",
                "log",
                "log",
            ),
            curve(
                "Bound fundamental TE mode",
                "x_um",
                [("mode", "normalized electric field")],
                "Transverse coordinate (micrometre)",
                "Normalized electric amplitude",
            ),
            curve(
                "Dielectric slab geometry",
                "x_um",
                [("refractive_index", "index")],
                "Transverse coordinate (micrometre)",
                "Refractive index",
            ),
        ],
        {
            "meshes": rows,
            "profile": [
                {"x_m": xx, "normalized_electric": field, "epsilon_r": eps}
                for xx, field, eps in zip(x, mode, epsilon, strict=True)
            ],
        },
        [
            "Finite-difference boundaries truncate the cladding; domain and mesh errors are distinct.",
            "Core-interface nodes use arithmetic half-cell permittivity for tangential electric fields.",
        ],
    )


def mode_perturbation(scale=1, seed=42):
    from .geometry import rectangle
    from .grid import from_bounds
    from .modes import cavity_modes, dielectric_shift
    from .results import curve, image, make

    grid = from_bounds((0, 0.02), (0, 0.012), 40 * scale + 1, 32 * scale + 1)
    patch = rectangle(grid, (0.007, 0.013), (0.004, 0.008)).astype(float)
    base = cavity_modes(grid, count=1)
    initial_frequency = base["cutoff_hz"][0]
    strengths = np.linspace(0, 0.5, 11)
    measured, predicted = [], []
    for strength in strengths:
        changed = cavity_modes(grid, 1 + strength * patch, count=1)
        measured.append(changed["cutoff_hz"][0] / initial_frequency - 1)
        predicted.append(dielectric_shift(grid, base["fields"][0], 1, strength * patch))
    measured, predicted = np.array(measured), np.array(predicted)
    return make(
        "mode-perturbation",
        {
            "domain_m": [0.02, 0.012],
            "grid_nodes": [grid.nx, grid.ny],
            "baseline_epsilon_r": 1,
            "maximum_patch_delta_epsilon": 0.5,
            "seed": seed,
        },
        {
            "baseline_frequency_hz": initial_frequency,
            "first_nonzero_prediction_relative_error": abs(predicted[1] / measured[1] - 1),
            "largest_absolute_fractional_shift_error": np.max(abs(predicted - measured)),
        },
        {
            "strength": strengths,
            "measured_percent": measured * 100,
            "predicted_percent": predicted * 100,
            "patch": patch * 0.5,
            "mode": base["fields"][0] / np.max(abs(base["fields"][0])),
        },
        [
            curve(
                "First-order shifts have a finite range of validity",
                "strength",
                [
                    ("measured_percent", "resolved eigenproblem"),
                    ("predicted_percent", "first-order prediction"),
                ],
                "Patch permittivity increment",
                "Frequency shift (%)",
            ),
            image("Maximum dielectric perturbation", "patch", grid, "Relative permittivity increment"),
            image("Unperturbed electric mode", "mode", grid, "Normalized Ez"),
        ],
        {
            "perturbations": [
                {"delta_epsilon": s, "measured_shift": a, "first_order_shift": b}
                for s, a, b in zip(strengths, measured, predicted, strict=True)
            ]
        },
        [
            "Perturbation theory uses the unperturbed mode and is only first order.",
            "The full eigenproblem is solved separately at each material contrast.",
        ],
    )
