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
