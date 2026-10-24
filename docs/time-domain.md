# Yee time-domain solvers

Yee1D advances Ey and Hz; TMz2D advances Ez,Hx,Hy. Material maps are positive scalar
permittivity, scalar permeability and nonnegative electric conductivity.
The magnetic update precedes the electric update. Conductivity is centered in time,
giving a trapezoidal electric damping coefficient rather than explicit unstable loss.

For minimum epsilon_r and mu_r, the multidimensional limit is
dt <= 1 / (v_max sqrt(sum(1/dx_i^2))). Every constructor validates the timestep.
Subcell anisotropy and full vector 3D material coupling are outside these solvers.

## Sources and captures

Current enters Ampere's law with physical SI units. The 1D sheet_current helper
divides a sheet current in A/m by dx. The 2D line_current helper divides A by dx dy.
A source cannot occupy a PEC node.

evolve returns electric and magnetic probe arrays, separate timestamps, snapshots,
snapshot step indices and final fields. Snapshot stride controls visualization
sampling only. Fourier analysis always uses the full-rate recorded probes.

The slab study integrates the device and a vacuum reference separately with the
same pulse and timestep. Reflected E is device E minus incident-reference E.
The transmitted/reference ratio is deembedded to the slab faces with the discrete
vacuum propagation constant. A declared incident-spectrum floor excludes frequencies
without adequate illumination; excluded ratios are never silently assigned a floor.

## Cavity ringdown

A source-free rectangular TM11 field initializes the cavity. An AR2 recurrence fit
extracts the isolated conjugate pole pair. The frequency is compared both with the
continuum rectangular formula and an independently derived discrete Yee recurrence.
The lossy study compares fitted decay and Q with the conductive discrete poles and
the continuum damped-oscillator result.

AR2 is an isolated-mode estimator. It is not a multimode spectral fitter, and a
superposition of close modes does not have a meaningful single fitted Q.

## Energy

The displayed electric-plus-magnetic snapshot energy uses staggered fields and
can oscillate numerically even in a lossless cavity. A separate modified leapfrog
energy uses the cross-time magnetic product and is invariant for the uniform
lossless PEC update. It is rejected for lossy or absorbing-boundary states.

Monitor frequency, mesh dispersion, record length, source bandwidth and boundary
return independently. Small update residuals alone do not establish continuum accuracy.
