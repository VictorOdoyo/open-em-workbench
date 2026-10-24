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
