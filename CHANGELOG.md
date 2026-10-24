# Changelog

## 0.1.0

Initial computational electromagnetics workbench:

- Conservative electrostatic finite volumes, full Maxwell capacitance matrices,
  floating conductors, work balance, material gradients and Maxwell stress.
- Filament and scalar-vector-potential magnetostatics, analytic loop fields,
  mutual inductance and fixed-current coil force.
- Passive material responses, normalized microwave networks, RLGC transmission
  lines, stable multilayer fields, absorption and periodic Bloch response.
- One-dimensional and scalar two-dimensional Yee solvers with conductivity,
  Mur/CPML boundaries, separate field timestamps and passive Debye ADE.
- TE/TM scalar cavity eigenmodes, slab roots/eigenfields, outgoing TMz FDFD and
  independent cylindrical harmonic scattering references.
- Constrained dielectric design, correlated tolerance propagation and mean bootstrap.
- Twenty-four reproducible studies, sealed numeric archives, CSV/JSON export,
  scientific figures, HTML reports, replay and integrity verification.
- Cross-platform test matrix, pinned CI actions and installed-wheel smoke checks.

The release is limited to its documented 1D/scalar 2D models. Geometry rasterization,
finite boundaries, time windows, cladding truncation and local optimization remain
explicit parts of result interpretation.
