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


def coupled_capacitors(scale=1, seed=42):
    from .capacitance import extract, floating_voltages, terminal_response
    from .electrostatics import energy
    from .geometry import conductor_labels, rectangle
    from .grid import Grid2D, from_bounds
    from .results import image, make

    grid = from_bounds((-0.02, 0.02), (-0.015, 0.015), 80 * scale + 1, 60 * scale + 1)
    masks = [
        rectangle(grid, (-0.008, -0.006), (-0.004, 0.004)),
        rectangle(grid, (0.006, 0.008), (-0.004, 0.004)),
    ]
    labels = conductor_labels(grid, masks)
    extracted = extract(grid, labels, 3)
    matrix = extracted["matrix"]
    voltage, charge = floating_voltages(matrix, [0, 1], [0, 1])
    potential = np.einsum("i,iyx->yx", voltage, extracted["basis"])
    _, terminal_energy = terminal_response(matrix, voltage)
    field_energy = energy(grid, potential, 3)
    matrix_plot = image("Maxwell capacitance matrix", "capacitance_pf_m", Grid2D(3, 3, 1, 1), "C (pF/m)")
    matrix_plot["xlabel"], matrix_plot["ylabel"] = "Driven conductor index", "Observed conductor index"
    matrix_plot["categorical"] = True
    rows = [{"observed": i, "driven": j, "capacitance_f_m": matrix[i, j]} for i in range(3) for j in range(3)]
    return make(
        "capacitance-coupling",
        {
            "domain_m": [0.04, 0.03],
            "grid_nodes": [grid.nx, grid.ny],
            "epsilon_r": 3,
            "driven_voltage_v": [0, 1],
            "floating_charge_c_m": 0,
            "seed": seed,
        },
        {
            "floating_voltage_v": voltage[2],
            "floating_charge_c_m": charge[2],
            "relative_reciprocity_defect": np.linalg.norm(matrix - matrix.T) / np.linalg.norm(matrix),
            "field_energy_j_m": field_energy,
            "terminal_energy_j_m": terminal_energy,
        },
        {
            "capacitance_pf_m": matrix * 1e12,
            "potential_v": potential,
            "unit_excitation_v": extracted["basis"][1],
            "conductor_labels": labels,
        },
        [
            matrix_plot,
            image(
                "One driven strip; second strip electrically floating", "potential_v", grid, "Potential (V)"
            ),
            image("Reference unit excitation for terminal one", "unit_excitation_v", grid, "Potential (V)"),
        ],
        {
            "capacitance": rows,
            "terminals": [
                {"terminal": i, "voltage_v": voltage[i], "charge_c_m": charge[i]} for i in range(3)
            ],
        },
        [
            "Two embedded rectangular strips inside a grounded box; translational invariance along z.",
            "All capacitances and energies are per metre; the floating strip is solved from zero net charge.",
        ],
    )


def coaxial_convergence(scale=1, seed=42):
    from .capacitance import coaxial
    from .electrostatics import conductor_charges, solve
    from .grid import from_bounds
    from .results import curve, image, make

    reference = coaxial(0.01, 0.03, 2.25)
    rows = []
    for base in (41, 65, 97):
        count = (base - 1) * scale + 1
        grid = from_bounds((-0.03, 0.03), (-0.03, 0.03), count, count)
        x, y = grid.mesh()
        radius = np.hypot(x, y)
        labels = np.full(grid.shape, -1, int)
        labels[radius >= 0.03] = 0
        labels[radius <= 0.01] = 1
        potential = solve(grid, 2.25, labels >= 0, (labels == 1).astype(float))["potential"]
        capacitance = conductor_charges(grid, labels, potential, 2.25)[1]
        rows.append(
            {
                "nodes_per_axis": count,
                "spacing_m": grid.dx,
                "capacitance_f_m": capacitance,
                "relative_error": abs(capacitance / reference - 1),
            }
        )
    return make(
        "coax-convergence",
        {
            "inner_radius_m": 0.01,
            "outer_radius_m": 0.03,
            "epsilon_r": 2.25,
            "mesh_nodes": [r["nodes_per_axis"] for r in rows],
            "seed": seed,
        },
        {
            "analytic_capacitance_f_m": reference,
            "finest_capacitance_f_m": rows[-1]["capacitance_f_m"],
            "finest_relative_error": rows[-1]["relative_error"],
        },
        {
            "spacing_m": [r["spacing_m"] for r in rows],
            "error": [r["relative_error"] for r in rows],
            "capacitance_pf_m": [r["capacitance_f_m"] * 1e12 for r in rows],
            "reference_pf_m": np.full(len(rows), reference * 1e12),
            "potential_v": potential,
        },
        [
            curve(
                "Curved boundary error is retained",
                "spacing_m",
                [("error", "relative error")],
                "Grid spacing (m)",
                "Capacitance relative error",
                "log",
                "log",
            ),
            curve(
                "Numerical and continuum capacitance",
                "spacing_m",
                [("capacitance_pf_m", "staircase geometry"), ("reference_pf_m", "exact coax")],
                "Grid spacing (m)",
                "Capacitance (pF/m)",
            ),
            image("Finest solved coaxial potential", "potential_v", grid, "Potential (V)"),
        ],
        {"meshes": rows},
        [
            "Circular conductors are rasterized at nodes; no curved-boundary correction is claimed.",
            "Refinement need not be monotonic when conductor coverage changes between meshes.",
        ],
    )
