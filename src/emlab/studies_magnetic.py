import numpy as np


def coil_mutual(scale=1, seed=42):
    from .grid import from_bounds
    from .magnetostatics import (
        circular_field,
        circular_loop,
        coaxial_force,
        coaxial_mutual,
        loop_axis,
        mutual_inductance,
        polyline_field,
    )
    from .results import curve, image, make

    exact = coaxial_mutual(0.1, 0.08, 0.05)
    rows = []
    for base in (16, 32, 64, 128):
        count = base * scale
        first = circular_loop(0.1, count)
        second = circular_loop(0.08, count, (0, 0, 0.05))
        numerical = mutual_inductance(first, second)
        rows.append(
            {"segments_per_coil": count, "mutual_h": numerical, "relative_error": abs(numerical / exact - 1)}
        )
    z_axis = np.linspace(-0.2, 0.2, 101)
    observation = np.column_stack((np.zeros_like(z_axis), np.zeros_like(z_axis), z_axis))
    polygon_axis = polyline_field(observation, circular_loop(0.1, 128 * scale))[:, 2]
    exact_axis = loop_axis(0.1, z_axis)
    grid = from_bounds((-0.18, 0.18), (-0.15, 0.15), 120 * scale + 1, 100 * scale + 1)
    x, z = grid.mesh()
    excluded = np.hypot(abs(x) - 0.1, z) < 0.004
    points = np.stack((x, np.zeros_like(x), z), axis=-1)
    field = np.zeros((*grid.shape, 3))
    field[~excluded] = circular_field(points[~excluded], 0.1)
    mapped = image(
        "Circular-loop axial field; filament vicinity excluded",
        "bz_ut",
        grid,
        "Bz (microtesla)",
        mask="excluded",
    )
    mapped["ylabel"] = "z (m)"
    return make(
        "coil-mutual",
        {
            "radius_first_m": 0.1,
            "radius_second_m": 0.08,
            "separation_m": 0.05,
            "currents_a": [1, 1],
            "quadrature_order": 4,
            "segments": [r["segments_per_coil"] for r in rows],
            "excluded_filament_radius_m": 0.004,
            "seed": seed,
        },
        {
            "analytic_mutual_h": exact,
            "finest_mutual_h": rows[-1]["mutual_h"],
            "finest_relative_error": rows[-1]["relative_error"],
            "coaxial_attractive_force_n": coaxial_force(0.1, 0.08, 0.05, 1, 1),
        },
        {
            "segments": [r["segments_per_coil"] for r in rows],
            "error": [r["relative_error"] for r in rows],
            "z_m": z_axis,
            "axis_polygon_ut": polygon_axis * 1e6,
            "axis_exact_ut": exact_axis * 1e6,
            "bz_ut": field[..., 2] * 1e6,
            "excluded": excluded,
        },
        [
            curve(
                "Neumann integral versus elliptic reference",
                "segments",
                [("error", "relative error")],
                "Polygon segments per coil",
                "Mutual-inductance relative error",
                "log",
                "log",
            ),
            curve(
                "Independent polygon and analytic axis fields",
                "z_m",
                [("axis_polygon_ut", "polygon filaments"), ("axis_exact_ut", "analytic loop")],
                "Axial position (m)",
                "Bz (microtesla)",
            ),
            mapped,
        ],
        {
            "mutual": rows,
            "axis": [
                {"z_m": z, "polygon_bz_t": b, "analytic_bz_t": a}
                for z, b, a in zip(z_axis, polygon_axis, exact_axis, strict=True)
            ],
        },
        [
            "Filament currents are magnetostatic; fields on the wire itself are singular.",
            "The map uses the analytic circular-loop field; convergence curves use independent polygon quadrature.",
        ],
    )


def current_fields(scale=1, seed=42):
    from .core import MU0
    from .geometry import circle
    from .grid import dual_weights, from_bounds, nearest_index
    from .magnetostatics import solve_az
    from .results import image, make

    grid = from_bounds((-0.06, 0.06), (-0.04, 0.04), 80 * scale + 1, 60 * scale + 1)
    positive = circle(grid, (-0.02, 0), 0.006)
    negative = circle(grid, (0.02, 0), 0.006)
    weights = dual_weights(grid)
    current = positive.astype(float) / np.sum(weights[positive]) - negative.astype(float) / np.sum(
        weights[negative]
    )
    result = solve_az(grid, current)
    bx, by = result["bx_t"], result["by_t"]
    y0, x0 = nearest_index(grid, -0.035, -0.015)
    y1, x1 = nearest_index(grid, -0.005, 0.015)
    circulation = (
        np.trapezoid(bx[y0, x0 : x1 + 1], grid.x[x0 : x1 + 1])
        + np.trapezoid(by[y0 : y1 + 1, x1], grid.y[y0 : y1 + 1])
        - np.trapezoid(bx[y1, x0 : x1 + 1], grid.x[x0 : x1 + 1])
        - np.trapezoid(by[y0 : y1 + 1, x0], grid.y[y0 : y1 + 1])
    ) / MU0
    source_work = 0.5 * np.sum(current * result["az_wb_m"] * weights)
    return make(
        "magnetostatic-current",
        {
            "domain_m": [0.12, 0.08],
            "grid_nodes": [grid.nx, grid.ny],
            "wire_radius_m": 0.006,
            "current_centers_m": [[-0.02, 0], [0.02, 0]],
            "currents_a": [1, -1],
            "mu_r": 1,
            "boundary_az_wb_m": 0,
            "seed": seed,
        },
        {
            "net_current_a": np.sum(current * weights),
            "ampere_current_a": circulation,
            "ampere_relative_error": abs(circulation - 1),
            "field_energy_j_m": result["energy_j_m"],
            "source_work_j_m": source_work,
            "solver_residual": result["residual"],
        },
        {
            "current_a_m2": current,
            "az_uwb_m": result["az_wb_m"] * 1e6,
            "b_ut": np.hypot(bx, by) * 1e6,
            "bx_t": bx,
            "by_t": by,
        },
        [
            image("Current and return conductor", "current_a_m2", grid, "Jz (A/m^2)"),
            image("Solved magnetic vector potential", "az_uwb_m", grid, "Az (microWb/m)"),
            image("Recovered magnetic flux density", "b_ut", grid, "|B| (microtesla)"),
        ],
        {
            "audit": [
                {
                    "field_energy_j_m": result["energy_j_m"],
                    "source_work_j_m": source_work,
                    "ampere_current_a": circulation,
                    "net_current_a": np.sum(current * weights),
                }
            ]
        },
        [
            "Two translationally invariant finite-area current distributions inside a finite Az=0 box.",
            "Currents are normalized by rasterized area; the nodal Ampere contour has its own quadrature error.",
        ],
    )
