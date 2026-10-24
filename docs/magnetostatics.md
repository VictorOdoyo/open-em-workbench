# Magnetostatics

magnetostatics implements finite straight-segment Biot-Savart fields, polyline coils,
circular-loop axis fields and off-axis elliptic-integral fields. Field evaluation on
an ideal filament is singular and is rejected. Plots mask a declared wire vicinity;
masking does not invent a finite conductor radius.

The exact finite-segment vector potential can be differentiated numerically to
check its curl against the independently evaluated magnetic field.

## Inductance and force

Mutual inductance uses the Neumann double line integral with Gauss quadrature along
each segment. Distinct nonintersecting filaments are required. Reciprocity, loop
orientation, polygon refinement and coaxial analytic mutual inductance are tested.

Self inductance is a separate thin-wire approximation requiring a physical wire
radius. Its internal-flux contribution depends on current distribution. The API
does not apply the singular mutual-filament formula to a loop and itself.

Coupled energy is one half I^T L I. The inductance matrix must be symmetric and
positive semidefinite. Coaxial fixed-current force is I1 I2 dM/dz, with separation
and current signs stated explicitly.

## Two-dimensional vector potential

For currents along z, solve -div((1/mu_r) grad(Az)) = MU0 Jz.
Then Bx = dAz/dy and By = -dAz/dx. The current-field study imposes Az=0 on
the finite exterior rectangle, normalizes each rasterized current disk to its
specified total current, and compares field energy with source work.

That exterior condition influences the field; it is not an infinite-space Green
boundary. Ampere circulation is evaluated around a contour enclosing one current
disk. Mesh, contour interpolation and exterior truncation have different errors.
