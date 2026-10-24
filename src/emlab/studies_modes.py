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
