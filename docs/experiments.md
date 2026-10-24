# Experiment catalog

Run any case with emlab run --case NAME --output outputs/NAME.
Defaults are resolution_scale=1 and seed=42. The scale increases sample counts while
preserving physical dimensions. Cost is dominated by sparse 2D field solves and
full time histories; scale 3 can require substantially more memory.

| Case | Calculation | Reference or diagnostic |
| --- | --- | --- |
| poisson-convergence | Three-grid Poisson solve | Manufactured sine field and fitted continuum order |
| dielectric-interface | Layered dielectric electrostatics | Constant D, discontinuous E, series capacitance |
| capacitance-coupling | Three-terminal Maxwell matrix | Reciprocity, floating charge, matrix/field energy |
| coax-convergence | Rasterized annular conductors | Analytic coax capacitance, honest geometry error |
| multilayer-coating | Quarter-wave antireflection film | Design-frequency Fresnel matching and R+T |
| bragg-mirror | Finite periodic dielectric reflector | Bloch attenuation and internal standing field |
| absorbing-stack | Passive two-layer slab | Boundary flux and independently integrated Joule loss |
| microwave-lines | Lossy coax and ideal transformer | RLGC passivity, delay and design match |
| cavity-cutoffs | Scalar TE/TM eigenmodes | Rectangular analytic cutoffs |
| slab-guidance | Transverse slab eigenproblem | Independent even/odd transcendental roots |
| mode-perturbation | Re-solved dielectric increments | First-order eigenfrequency shift |
| coil-mutual | Polygon coils and Neumann quadrature | Elliptic mutual inductance and axis field |
| magnetostatic-current | Scalar Az for two current disks | Ampere circulation and source/field energy |
| fdtd-slab | Two independent 1D Yee integrations | Complex deembedded multilayer scattering |
| fdtd-debye | Passive auxiliary-polarization integration | Continuum and discrete Debye response |
| fdtd-cavity | Source-free 2D TM11 ringdown | Continuum/discrete frequency and leapfrog invariant |
| fdtd-lossy-cavity | Conductive 2D cavity decay | Discrete pole Q versus fitted Q |
| fdtd-absorber | PEC, Mur and CPML packet returns | Measured late return and PML-thickness sweep |
| fdfd-line-source | Outgoing stretched-coordinate field | Complex Hankel Green solution |
| cylinder-scattering | Three rasterized PEC meshes | Independent harmonic-series exterior field |
| dielectric-sensitivity | Material derivative at selected nodes | Independently re-solved central differences |
| maxwell-force | Closed stress contour | Grounded image-line attraction |
| dielectric-design | Bounded material-budget optimization | Accepted energy increase and budget conservation |
| manufacturing-uncertainty | Correlated coax tolerance samples | Linearized covariance and iid mean bootstrap |

The [measured gallery](results.md) contains complete default captures with plots,
CSV tables, setup parameters and source provenance. Every report states its own
geometry, normalization, assumptions and masks.

A conservation defect, algebraic residual, discretization error and statistical
confidence interval answer different questions. Compare like quantities when
interpreting the summary metrics. Gallery values are measured outputs from the
archived run, rather than thresholds copied from the tests.
