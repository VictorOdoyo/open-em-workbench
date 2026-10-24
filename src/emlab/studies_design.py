import numpy as np


def sensitivity(scale=1, seed=42):
    from .electrostatics import energy, energy_sensitivity, solve
    from .grid import from_bounds
    from .results import curve, image, make

    grid = from_bounds((0, 0.04), (0, 0.03), 40 * scale + 1, 30 * scale + 1)
    x, y = grid.mesh()
    epsilon = 2.5 + 0.5 * np.sin(np.pi * x / 0.04) * np.sin(np.pi * y / 0.03)
    voltage = x / 0.04 + 0.3 * np.sin(np.pi * y / 0.03)
    potential = solve(grid, epsilon, voltage=voltage)["potential"]
    derivative = energy_sensitivity(grid, potential, epsilon)
    rows = []
    step = 1e-4
    for iy, ix in [(5, 5), (10, 15), (15, 20), (20, 30), (25, 35), (10, 32)]:
        iy, ix = iy * scale, ix * scale
        energies = []
        for sign in (-1, 1):
            changed = epsilon.copy()
            changed[iy, ix] += sign * step
            field = solve(grid, changed, voltage=voltage)["potential"]
            energies.append(energy(grid, field, changed))
        measured = (energies[1] - energies[0]) / (2 * step)
        rows.append(
            {
                "x_m": grid.x[ix],
                "y_m": grid.y[iy],
                "analytic_j_m": derivative[iy, ix],
                "finite_difference_j_m": measured,
                "relative_error": abs(measured - derivative[iy, ix]) / abs(derivative[iy, ix]),
            }
        )
    return make(
        "dielectric-sensitivity",
        {
            "domain_m": [0.04, 0.03],
            "grid_nodes": [grid.nx, grid.ny],
            "perturbation_epsilon_r": step,
            "seed": seed,
        },
        {"maximum_relative_error": max(r["relative_error"] for r in rows)},
        {
            "sample": np.arange(1, len(rows) + 1),
            "analytic": [r["analytic_j_m"] for r in rows],
            "measured": [r["finite_difference_j_m"] for r in rows],
            "sensitivity": derivative,
            "potential": potential,
        },
        [
            curve(
                "Stationary-energy derivative versus independent re-solves",
                "sample",
                [("analytic", "exact discrete derivative"), ("measured", "central difference")],
                "Sampled node",
                "dW / d epsilon_r (J/m)",
            ),
            image("Spatial material sensitivity", "sensitivity", grid, "dW / d epsilon_r (J/m)"),
        ],
        {"samples": rows},
        [
            "Fixed conductor voltages and no free charge define the stationary energy derivative.",
            "This verifies the derivative of the discrete harmonic-edge material operator.",
        ],
    )


def force(scale=1, seed=42):
    from .core import EPS0
    from .electrostatics import maxwell_stress, rectangular_force
    from .grid import from_bounds
    from .results import curve, image, make
    from .statics import line_charge

    charge, height = 1e-9, 0.1
    reference = -(charge**2) / (4 * np.pi * EPS0 * height)
    rows = []
    for intervals in (40, 80, 160):
        count = intervals * scale + 1
        grid = from_bounds((-0.1, 0.1), (0, 0.2), count, count)
        x, y = grid.mesh()
        points = np.stack((x, y), axis=-1)
        mask = np.hypot(x, y - height) < grid.dx / 2
        electric = np.zeros((*grid.shape, 2))
        electric[~mask] = (
            line_charge(points[~mask], charge, (0, height))[1]
            + line_charge(points[~mask], -charge, (0, -height))[1]
        )
        quarter = (count - 1) // 4
        measured = rectangular_force(
            grid,
            maxwell_stress(electric[..., 0], electric[..., 1]),
            (quarter, 3 * quarter, quarter, 3 * quarter),
        )
        rows.append(
            {
                "spacing_m": grid.dx,
                "force_x_n_m": measured[0],
                "force_y_n_m": measured[1],
                "relative_error": abs(measured[1] - reference) / abs(reference),
            }
        )
    return make(
        "maxwell-force",
        {
            "line_charge_c_m": charge,
            "height_m": height,
            "contour_m": [-0.05, 0.05, 0.05, 0.15],
            "mesh_nodes": [(n * scale + 1) for n in (40, 80, 160)],
            "seed": seed,
        },
        {
            "reference_force_y_n_m": reference,
            "finest_force_y_n_m": rows[-1]["force_y_n_m"],
            "finest_relative_error": rows[-1]["relative_error"],
        },
        {
            "spacing_mm": [r["spacing_m"] * 1e3 for r in rows],
            "relative_error": [r["relative_error"] for r in rows],
            "electric_magnitude": np.linalg.norm(electric, axis=-1),
            "mask": mask,
        },
        [
            curve(
                "Stress contour quadrature converges to image-line force",
                "spacing_mm",
                [("relative_error", "force relative error")],
                "Grid spacing (mm)",
                "Relative error",
                "log",
                "log",
            ),
            image(
                "Image-line electric field above grounded plane",
                "electric_magnitude",
                grid,
                "|E| (V/m)",
                mask="mask",
            ),
        ],
        {"meshes": rows},
        [
            "The field is analytic; this isolates closed-contour stress quadrature error.",
            "The singular filament node is masked and never lies on the integration contour.",
            "Force is per unit out-of-plane length.",
        ],
    )


def design(scale=1, seed=42):
    from .design import optimize_dielectric
    from .geometry import conductor_labels, rectangle
    from .grid import dual_weights, from_bounds
    from .results import curve, image, make

    grid = from_bounds((0, 0.04), (0, 0.03), 40 * scale + 1, 30 * scale + 1)
    electrode = rectangle(grid, (0.018, 0.022), (0.01, 0.02))
    labels = conductor_labels(grid, [electrode])
    mask = labels < 0
    result = optimize_dielectric(grid, labels >= 0, (labels == 1).astype(float), mask, iterations=16)
    weights = dual_weights(grid)[mask]
    final_mean = float(np.sum(result["epsilon_r"][mask] * weights) / weights.sum())
    history = result["history"]
    initial, final = history[0]["energy_j_m"], history[-1]["energy_j_m"]
    return make(
        "dielectric-design",
        {
            "domain_m": [0.04, 0.03],
            "grid_nodes": [grid.nx, grid.ny],
            "voltage_v": 1,
            "initial_epsilon_r": 2.5,
            "bounds_epsilon_r": [1, 4],
            "maximum_iterations": 16,
            "initial_step": 0.5,
            "seed": seed,
        },
        {
            "initial_capacitance_f_m": 2 * initial,
            "final_capacitance_f_m": 2 * final,
            "capacitance_gain_fraction": final / initial - 1,
            "material_mean_defect": abs(final_mean - result["material_mean"]),
            "accepted_iterations": len(history) - 1,
        },
        {
            "iteration": [r["iteration"] for r in history],
            "energy_pj_m": [r["energy_j_m"] * 1e12 for r in history],
            "epsilon_r": result["epsilon_r"],
            "potential": result["potential"],
            "design_mask": mask,
        },
        [
            curve(
                "Fixed-voltage energy with conserved material",
                "iteration",
                [("energy_pj_m", "accepted objective")],
                "Accepted iteration",
                "Stored energy (pJ/m)",
            ),
            image("Final relative permittivity", "epsilon_r", grid, "Relative permittivity"),
            image("Final potential", "potential", grid, "Potential (V)"),
        ],
        {"history": history},
        [
            "This is bounded local optimization of a discrete nodal dielectric field.",
            "The weighted mean permittivity is fixed; no global optimum or manufacturability is claimed.",
            "Capacitance per unit length equals twice stored energy at the imposed one volt.",
        ],
    )


def uncertainty(scale=1, seed=42):
    from .core import EPS0, random_stream
    from .results import curve, make
    from .uncertainty import linearized_covariance, mean_bootstrap

    nominal = np.array([0.001, 0.003, 2.25])
    sigma = np.array([10e-6, 15e-6, 0.03])
    correlation = np.array([[1, 0.3, 0], [0.3, 1, 0], [0, 0, 1]])
    covariance = sigma[:, None] * correlation * sigma[None, :]
    rng = random_stream(seed, "coax-manufacturing")
    samples = rng.multivariate_normal(nominal, covariance, size=512 * scale)
    a, b, epsilon = samples.T
    if np.any((a <= 0) | (b <= a) | (epsilon <= 0)):
        raise ValueError("assumed normal tolerance draw produced nonphysical geometry")
    capacitance = 2 * np.pi * EPS0 * epsilon / np.log(b / a)
    nominal_c = 2 * np.pi * EPS0 * nominal[2] / np.log(nominal[1] / nominal[0])
    logarithm = np.log(nominal[1] / nominal[0])
    jacobian = nominal_c * np.array(
        [1 / (nominal[0] * logarithm), -1 / (nominal[1] * logarithm), 1 / nominal[2]]
    )
    predicted = float(np.sqrt(linearized_covariance(jacobian, covariance)[0, 0]))
    measured = float(np.std(capacitance, ddof=1))
    mean_interval = mean_bootstrap(capacitance, seed=seed)
    population_interval = np.quantile(capacitance, [0.025, 0.975])
    histogram, edges = np.histogram(capacitance * 1e12, bins=25, density=True)
    center = (edges[:-1] + edges[1:]) / 2
    gaussian = np.exp(-0.5 * ((center - nominal_c * 1e12) / (predicted * 1e12)) ** 2) / (
        predicted * 1e12 * np.sqrt(2 * np.pi)
    )
    rows = [
        {"inner_radius_m": ia, "outer_radius_m": ib, "epsilon_r": ie, "capacitance_f_m": ic}
        for ia, ib, ie, ic in zip(a, b, epsilon, capacitance, strict=True)
    ]
    return make(
        "manufacturing-uncertainty",
        {
            "nominal_inner_radius_m": nominal[0],
            "nominal_outer_radius_m": nominal[1],
            "nominal_epsilon_r": nominal[2],
            "parameter_standard_deviations": sigma,
            "correlation_matrix": correlation,
            "sample_count": len(samples),
            "bootstrap_resamples": 2000,
            "seed": seed,
        },
        {
            "nominal_capacitance_f_m": nominal_c,
            "sample_mean_f_m": float(np.mean(capacitance)),
            "linearized_std_f_m": predicted,
            "sample_std_f_m": measured,
            "std_relative_difference": abs(measured - predicted) / predicted,
            "mean_ci95_low_f_m": mean_interval[0],
            "mean_ci95_high_f_m": mean_interval[1],
            "population_p025_f_m": population_interval[0],
            "population_p975_f_m": population_interval[1],
        },
        {
            "capacitance_pf_m": center,
            "density": histogram,
            "linearized_density": gaussian,
            "samples": samples,
            "sample_capacitance_f_m": capacitance,
        },
        [
            curve(
                "Assumed manufacturing population",
                "capacitance_pf_m",
                [("density", "Monte Carlo density"), ("linearized_density", "linearized normal density")],
                "Capacitance (pF/m)",
                "Probability density (m/pF)",
            )
        ],
        {"samples": rows},
        [
            "These are assumed correlated normal tolerances, not measured manufacturing data.",
            "Independent draws represent a population; the percentile bootstrap interval estimates its mean.",
            "Population percentiles describe unit-to-unit variation and are not confidence limits on the mean.",
        ],
    )
