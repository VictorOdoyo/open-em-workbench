# Frequency-domain fields and scattering

fdfd constructs a scalar TMz Helmholtz operator using complex coordinate stretching.
The matrix is complex symmetric rather than Hermitian. A direct sparse solve
reports its residual. A low residual establishes algebraic solution quality only.

The positive-time convention makes the free-space outgoing Green function
G(r) = -i H0^(2)(k r)/4.
For a z-directed line current I, Ez = -i omega MU0 I G. The line-source study
compares complex sampled fields, not only magnitudes, so a radiation-sign error
cannot pass unnoticed. Singular cells and PML are explicitly masked in figures.

## Incident and scattered fields

A discrete incident plane wave solves the grid dispersion relation along the
specified angle. This prevents a spurious bulk residual from being interpreted as
an object source. A continuum Mie reference still has continuum dispersion;
the difference belongs in the discretization error.

PEC scattering imposes Escattered=-Eincident on the obstacle. Dielectric scattering
uses the permittivity contrast source. Contrast must remain outside the absorbing
layer. The finite computational exterior has PEC termination beyond the PML.

## Infinite-cylinder reference

scattering independently evaluates PEC and passive dielectric cylinder harmonic
coefficients with outgoing Hankel functions. The dielectric calculation scales
internal Bessel functions to avoid unnecessary exponential overflow.

The cylinder study compares numerical complex near fields at sixteen exterior
angles across three meshes. It retains the nonmonotonic error introduced by
rasterized circles. The shown differential width is from the analytic harmonic
series; there is no claimed numerical near-to-far transform in that figure.

Integrated scattering and extinction widths obey the optical theorem; their
difference is absorption for passive materials. Cylinder widths have units metres
per out-of-plane length, and differential width uses m/rad.

Change physical boundary distance, PML thickness, mesh and harmonic truncation
independently for a new frequency or object size.

Reference special function:
[SciPy Hankel2](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.hankel2.html).
