# Open EM Workbench

[![CI](https://github.com/VictorOdoyo/open-em-workbench/actions/workflows/ci.yml/badge.svg)](https://github.com/VictorOdoyo/open-em-workbench/actions/workflows/ci.yml)

A Python workbench for computational electromagnetics: conservative electrostatics,
magnetostatics, microwave networks, multilayer optics, waveguide modes, Yee FDTD,
frequency-domain scattering, and material design.

The field solvers and analytical references are implemented in this repository using
NumPy and SciPy. Twenty-four runnable studies compare independent formulations,
report conservation defects, and retain numeric fields alongside scientific figures.

**[Browse the measured study gallery](docs/results.md)** ·
[Methods and conventions](docs/conventions.md) · [Experiment catalog](docs/experiments.md) ·
[Reproduce and replay](docs/reproducibility.md)

## Run a study

Python 3.11 or newer is required. From a checkout:

```bash
python -m pip install -e ".[dev]"
emlab list
emlab run --case fdtd-slab --output outputs/slab
emlab verify outputs/slab
```

For the locked development environment:

```bash
uv sync --extra dev --locked
uv run emlab run --config examples/coating.json --output outputs/coating
uv run pytest -q
uv run ruff check .
```

A report contains an HTML page, figure, JSON parameters, CSV tables, a sealed
pickle-free NPZ archive, source/environment provenance, and a SHA-256 manifest.

```bash
emlab batch --data-only --output outputs/catalog
emlab render outputs/catalog
emlab verify outputs/catalog
emlab replay outputs/slab/fields.npz --output outputs/slab-replayed
```

Replay preserves the original solver provenance. It does not run the physics again.

## What is implemented

| Area | Numerical methods | Independent checks |
| --- | --- | --- |
| Electrostatics | Harmonic-edge finite volumes, sparse Dirichlet solve, Maxwell capacitance matrices | Manufactured solution, coax capacitance, charge/work/energy, reciprocity |
| Magnetostatics | Vector-potential solve, finite wire Biot-Savart, Neumann quadrature | Elliptic loop fields, analytic mutual inductance, Ampere circulation |
| Waves and RF | Power-wave S parameters, RLGC lines, stable layer scattering | Fresnel limits, passivity, impedance matching, volume Joule absorption |
| Time domain | 1D Yee, scalar 2D TMz Yee, Mur, CPML, conductive loss, Debye ADE | Two-run slab scattering, cavity frequencies/Q, exact discrete energy |
| Frequency domain | Stretched-coordinate scalar TMz Helmholtz, PEC/dielectric contrast scattering | Outgoing Hankel Green function, cylindrical harmonic series, optical theorem |
| Modes and design | TE/TM scalar eigenproblems, slab modes, energy gradients, constrained material ascent | Rectangular cutoffs, transcendental slab roots, resolved finite differences |
| Uncertainty | Correlated linearized covariance, Monte Carlo, mean bootstrap | Analytic coax derivatives and separate population/mean intervals |

## Interpretation

All physical quantities use SI units. Phasors use **exp(+i omega t)**, outgoing waves
use **exp(-i k r)**, and passive permittivity has negative imaginary part.
Cartesian arrays use **[y, x]**. Time-domain electric and magnetic traces have
separate timestamps because the Yee fields are staggered.

This is a research and education codebase for 1D and scalar 2D models. It does not
implement a general 3D vector solver, a complete CAD mesher, or hardware certification.
Rasterized curved conductors, finite PMLs, finite capture windows, finite cladding,
and local material optimization have explicit limitations in each report.

## Read and extend

- [Discrete operators](docs/operators.md), [electrostatics](docs/electrostatics.md), [magnetostatics](docs/magnetostatics.md)
- [Waves and networks](docs/waves.md), [time domain](docs/time-domain.md), [absorbers](docs/absorbers.md)
- [Dispersive materials](docs/dispersion.md), [modes](docs/modes.md), [scattering](docs/scattering.md)
- [Sensitivity and design](docs/design.md), [uncertainty](docs/uncertainty.md), [Python API](docs/api.md)
- [References](docs/references.md), [contributing](CONTRIBUTING.md), [release notes](CHANGELOG.md)

MIT licensed. Numerical results are simulations with stated model assumptions.
