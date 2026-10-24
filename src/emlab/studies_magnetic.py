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
