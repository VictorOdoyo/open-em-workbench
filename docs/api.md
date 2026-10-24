# Python API examples

The command catalog provides fixed, documented engineering studies. For a new
geometry, call the numerical modules directly and make a result with explicit units.

```python
from emlab.grid import from_bounds
from emlab.electrostatics import solve, energy
grid = from_bounds((0, 0.04), (0, 0.03), 81, 61)
x, y = grid.mesh()
field = solve(grid, epsilon_r=2.5, voltage=x / 0.04)
print(energy(grid, field["potential"], 2.5))  # J/m
```

Default electrostatic boundaries are Dirichlet on all exterior nodes; the supplied
voltage array sets their values. Interior entries of that array are ignored unless
explicitly included in a fixed mask.

[custom_capacitor.py](../examples/custom_capacitor.py) builds a grounded enclosure
and an embedded rectangular electrode, extracts the Maxwell matrix and saves a
field report. Run it from the repository root:

```bash
uv run python examples/custom_capacitor.py
uv run emlab verify outputs/custom-capacitor
```

For a catalog study from Python:

```python
from emlab.experiments import ExperimentConfig, run
from emlab.reporting import write_report
result = run(ExperimentConfig("fdtd-cavity", resolution_scale=1, seed=42))
write_report(result, "outputs/cavity")
```

## Module map

| Modules | Responsibility |
| --- | --- |
| core, materials | SI constants, validation, passive constitutive responses |
| grid, geometry, operators | Cartesian coordinates, raster masks, sparse conservative graph |
| electrostatics, capacitance, statics | Poisson fields, terminal models, analytic electrostatic references |
| magnetostatics | Wires, loops, mutual inductance and scalar vector-potential solve |
| waves, networks, transmission, layers | Interfaces, normalized S matrices, lines and stable layered fields |
| timedomain, fdtd1d, fdtd2d, sources, monitors | CFL, Yee/CPML/ADE updates, physical currents and Fourier/ringdown analysis |
| modes, fdfd, scattering | Eigenproblems, outgoing frequency-domain fields and harmonic references |
| design, uncertainty | Material gradients/projection and stochastic propagation |
| results, reporting, experiments, batch | Validated datasets, sealed reports, strict requests and replay |

Lower-level numerical functions expect arrays of the documented shape. Field
solvers accept explicit material maps; a catalog request intentionally exposes only
case, resolution and seed. Modify a physical study or use the API for new dimensions,
materials or frequencies, and capture the new physical parameters in its result.
