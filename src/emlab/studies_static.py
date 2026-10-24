import numpy as np


def poisson(scale=1, seed=42):
    from .core import convergence_order, relative_error
    from .electrostatics import manufactured_sine, solve
    from .grid import from_bounds
    from .results import curve, image, make

    rows = []
    for base in (17, 33, 65):
        count = (base - 1) * scale + 1
        grid = from_bounds((0, 1), (0, 1), count, count)
        exact, density = manufactured_sine(grid, 3)
        solved = solve(grid, 3, voltage=exact, charge_density=density)
        numerical = solved["potential"]
        rows.append(
            {
                "nodes_per_axis": count,
                "spacing_m": grid.dx,
                "relative_l2_error": relative_error(exact, numerical),
                "solver_residual": solved["residual"],
            }
        )
    spacing = np.array([r["spacing_m"] for r in rows])
    errors = np.array([r["relative_l2_error"] for r in rows])
    order, _ = convergence_order(spacing, errors)
    return make(
        "poisson-convergence",
        {
            "domain_m": [1, 1],
            "epsilon_r": 3,
            "mesh_nodes": [r["nodes_per_axis"] for r in rows],
            "source": "manufactured sin(pi x) sin(pi y)",
            "seed": seed,
        },
        {
            "observed_order": order,
            "finest_relative_error": errors[-1],
            "maximum_solver_residual": max(r["solver_residual"] for r in rows),
        },
        {
            "spacing_m": spacing,
            "relative_error": errors,
            "numerical": numerical,
            "exact": exact,
            "error_v": numerical - exact,
        },
        [
            curve(
                "Continuum error under mesh refinement",
                "spacing_m",
                [("relative_error", "numerical")],
                "Grid spacing (m)",
                "Relative L2 potential error",
                "log",
                "log",
            ),
            image("Finest solved potential", "numerical", grid, "Potential (V)"),
            image("Numerical minus manufactured potential", "error_v", grid, "Potential error (V)"),
        ],
        {"meshes": rows},
        [
            "Uniform scalar dielectric; Dirichlet data on all four sides.",
            "The source is derived from the continuum PDE, not the discrete operator.",
        ],
    )


def dielectric_interface(scale=1, seed=42):
    from .core import EPS0
    from .electrostatics import conductor_charges, displacement, electric_field, energy, solve
    from .grid import from_bounds
    from .operators import harmonic_edges
    from .results import curve, image, make

    grid = from_bounds((0, 0.04), (0, 0.01), 80 * scale + 1, 24 * scale + 1)
    epsilon = np.where(grid.mesh()[0] <= 0.02, 2.0, 8.0)
    labels = np.full(grid.shape, -1, int)
    labels[:, 0], labels[:, -1] = 0, 1
    voltage = (labels == 0).astype(float)
    potential = solve(grid, epsilon, labels >= 0, voltage)["potential"]
    ex, _ = electric_field(grid, potential)
    dx, _ = displacement(grid, potential, epsilon)
    row = grid.ny // 2
    faces = harmonic_edges(grid, epsilon)[: ex.size].reshape(ex.shape)
    exact_c = EPS0 * 0.01 / (grid.dx * np.sum(1 / faces[row]))
    charge = conductor_charges(grid, labels, potential, epsilon)[0]
    variation = float(np.ptp(dx[row]) / np.mean(dx[row]))
    midpoint = (grid.x[:-1] + grid.x[1:]) / 2
    tables = [
        {"x_m": x, "electric_v_m": e, "displacement_c_m2": d, "epsilon_face": eps}
        for x, e, d, eps in zip(midpoint, ex[row], dx[row], faces[row], strict=True)
    ]
    return make(
        "dielectric-interface",
        {
            "domain_m": [0.04, 0.01],
            "grid_nodes": [grid.nx, grid.ny],
            "dielectrics": [2, 8],
            "voltage_v": 1,
            "seed": seed,
        },
        {
            "relative_displacement_variation": variation,
            "charge_capacitance_f_m": charge,
            "energy_capacitance_f_m": 2 * energy(grid, potential, epsilon),
            "discrete_series_reference_f_m": exact_c,
        },
        {
            "x_face_m": midpoint,
            "electric_x_v_m": ex[row],
            "displacement_x_c_m2": dx[row],
            "epsilon_r": epsilon,
            "potential_v": potential,
        },
        [
            curve(
                "Normal electric field changes with dielectric",
                "x_face_m",
                [("electric_x_v_m", "electric field")],
                "x (m)",
                "Ex (V/m)",
            ),
            curve(
                "Shared face displacement is conserved",
                "x_face_m",
                [("displacement_x_c_m2", "displacement")],
                "x (m)",
                "Dx (C/m^2)",
            ),
            image("Piecewise dielectric map", "epsilon_r", grid, "Relative permittivity"),
            image("Solved potential", "potential_v", grid, "Potential (V)"),
        ],
        {"interface": tables},
        [
            "Left/right electrodes; top/bottom have zero exterior normal flux.",
            "The exact series reference uses the same half-cell interface placement.",
        ],
    )
