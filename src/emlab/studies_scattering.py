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


def cylinder(scale=1, seed=42):
    from .core import relative_error
    from .fdfd import scatter
    from .geometry import circle
    from .grid import from_bounds, sample_bilinear
    from .results import curve, image, make
    from .scattering import exterior_field, pec_coefficients, widths

    frequency, radius = 0.8e9, 0.04
    angles = np.linspace(0, 2 * np.pi, 17)[:-1]
    probes = 0.09 * np.column_stack((np.cos(angles), np.sin(angles)))
    coefficients = pec_coefficients(frequency, radius)
    reference = exterior_field(coefficients, probes)["scattered"]
    rows = []
    for base in (81, 121, 161):
        count = (base - 1) * scale + 1
        grid = from_bounds((-0.2, 0.2), (-0.2, 0.2), count, count)
        pml_cells = max(2, round(0.056 / grid.dx))
        obstacle = circle(grid, (0, 0), radius)
        solved = scatter(grid, frequency, pec_mask=obstacle, pml_cells=pml_cells)
        numerical = np.array([sample_bilinear(grid, solved["scattered"], x, y) for x, y in probes])
        rows.append(
            {
                "nodes_per_axis": count,
                "spacing_m": grid.dx,
                "pml_cells": pml_cells,
                "complex_relative_l2": relative_error(reference, numerical),
                "solver_residual": solved["residual"],
            }
        )
    x, y = grid.mesh()
    masked = obstacle.copy()
    masked[:pml_cells], masked[-pml_cells:] = True, True
    masked[:, :pml_cells], masked[:, -pml_cells:] = True, True
    exact_map = np.zeros(grid.shape, complex)
    points = np.stack((x, y), axis=-1)
    exact_map[~masked] = exterior_field(coefficients, points[~masked])["total"]
    angular = np.linspace(0, 2 * np.pi, 361)
    width = widths(coefficients, angular)
    return make(
        "cylinder-scattering",
        {
            "frequency_hz": frequency,
            "radius_m": radius,
            "domain_m": [0.4, 0.4],
            "incident_peak_e_v_m": 1,
            "incidence_direction": "+x",
            "pml_physical_thickness_m": 0.056,
            "mesh_nodes": [r["nodes_per_axis"] for r in rows],
            "probe_radius_m": 0.09,
            "harmonic_maximum_order": int(np.max(abs(coefficients["orders"]))),
            "seed": seed,
        },
        {
            "finest_complex_relative_l2": rows[-1]["complex_relative_l2"],
            "maximum_solver_residual": max(r["solver_residual"] for r in rows),
            "analytic_scattering_width_m": width["scattering_m"],
            "analytic_optical_theorem_defect_m": abs(width["absorption_m"]),
        },
        {
            "spacing_mm": [r["spacing_m"] * 1e3 for r in rows],
            "relative_error": [r["complex_relative_l2"] for r in rows],
            "electric_total": solved["total"],
            "electric_scattered": solved["scattered"],
            "electric_exact": exact_map,
            "electric_error": abs(solved["total"] - exact_map),
            "mask": masked,
            "angle_degrees": angular * 180 / np.pi,
            "differential_width": width["differential_m_rad"],
        },
        [
            curve(
                "Sampled scattered-field error retains staircase geometry effects",
                "spacing_mm",
                [("relative_error", "relative complex L2 error")],
                "Grid spacing (mm)",
                "Relative error",
                "log",
                "log",
            ),
            curve(
                "Independent cylindrical-series angular width",
                "angle_degrees",
                [("differential_width", "analytic differential width")],
                "Scattering angle (degree)",
                "Scattering width (m/rad)",
            ),
            image(
                "Finest total field; PEC and PML masked", "electric_total", grid, "Re Ez (V/m)", mask="mask"
            ),
            image(
                "Absolute total-field error in the physical exterior",
                "electric_error",
                grid,
                "|E numerical - E exact| (V/m)",
                "abs",
                "mask",
            ),
        ],
        {
            "meshes": rows,
            "angular": [
                {"angle_rad": angle, "differential_width_m_rad": value}
                for angle, value in zip(angular, width["differential_m_rad"], strict=True)
            ],
        },
        [
            "The cylinder is infinitely long and perfectly conducting; scalar TMz polarization is used.",
            "Rasterized PEC boundaries converge separately from the centered-difference bulk stencil.",
            "The plotted angular width comes from the analytic series, not a claimed numerical near-to-far transform.",
            "Numerical plane waves satisfy grid dispersion; the reference uses continuum dispersion.",
        ],
    )
