import numpy as np


def line_source(scale=1, seed=42):
    from .core import MU0, relative_error
    from .fdfd import green_2d, solve
    from .grid import from_bounds
    from .results import curve, image, make

    frequency, current_a = 1e9, 0.001
    grid = from_bounds((-0.2, 0.2), (-0.2, 0.2), 100 * scale + 1, 100 * scale + 1)
    center = grid.nx // 2
    current = np.zeros(grid.shape)
    current[center, center] = current_a / (grid.dx * grid.dy)
    solved = solve(grid, frequency, current, pml_cells=14 * scale)
    positions = center + np.array([10, 15, 20, 25, 30]) * scale
    radius = grid.x[positions]
    exact = -2j * np.pi * frequency * MU0 * current_a * green_2d(frequency, radius)
    numerical = solved["electric"][center, positions]
    x, y = grid.mesh()
    excluded = np.hypot(x, y) < 2 * grid.dx
    pml = np.zeros(grid.shape, bool)
    cells = 14 * scale
    pml[:cells], pml[-cells:] = True, True
    pml[:, :cells], pml[:, -cells:] = True, True
    mask = excluded | pml
    rows = [
        {
            "radius_m": r,
            "numerical_real_v_m": n.real,
            "numerical_imag_v_m": n.imag,
            "reference_real_v_m": e.real,
            "reference_imag_v_m": e.imag,
            "relative_amplitude_error": abs(n - e) / abs(e),
        }
        for r, n, e in zip(radius, numerical, exact, strict=True)
    ]
    return make(
        "fdfd-line-source",
        {
            "frequency_hz": frequency,
            "line_current_a": current_a,
            "domain_m": [0.4, 0.4],
            "grid_nodes": [grid.nx, grid.ny],
            "pml_cells": cells,
            "pml_target_reflection": 1e-8,
            "pml_kappa": 1,
            "pml_alpha_s": 0,
            "seed": seed,
        },
        {
            "complex_field_relative_l2": relative_error(exact, numerical),
            "solver_residual": solved["residual"],
        },
        {
            "radius_m": radius,
            "numerical_magnitude": abs(numerical),
            "reference_magnitude": abs(exact),
            "point_error": abs(numerical - exact) / abs(exact),
            "electric": solved["electric"],
            "mask": mask,
        },
        [
            curve(
                "Outgoing source-field magnitude",
                "radius_m",
                [("numerical_magnitude", "finite difference"), ("reference_magnitude", "Hankel reference")],
                "Observation radius (m)",
                "|Ez| (V/m)",
            ),
            curve(
                "Complex field error includes amplitude and phase",
                "radius_m",
                [("point_error", "relative complex error")],
                "Observation radius (m)",
                "Relative error",
            ),
            image("Outgoing real electric phasor", "electric", grid, "Re Ez (V/m)", mask="mask"),
            image("Outgoing field magnitude", "electric", grid, "|Ez| (V/m)", "abs", "mask"),
        ],
        {"radial": rows},
        [
            "A physical z-directed line current is represented by one finite grid cell.",
            "The singular source neighbourhood and PML are masked in plots; all numeric fields remain in the archive.",
            "Point-source Green comparison uses only nonsingular samples in the physical interior.",
        ],
    )
