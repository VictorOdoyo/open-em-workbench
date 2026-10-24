# Electrostatics and capacitance

electrostatics.solve solves -div(epsilon grad(phi)) = rho/EPS0 with specified
conductor potentials. Use electric_field for edge fields and displacement for
conservative D. nodal_field is an interpolated display/traction field.

The manufactured sine study prescribes an analytic potential and its continuum
source. Three grids distinguish discretization error from the sparse linear residual.
The dielectric-interface study checks constant normal D and the jump in E.

## Maxwell capacitance matrix

capacitance.extract excites one conductor at a time while holding the others at
zero. All basis fields share a sparse LU factorization. Integrated reactions form
the full matrix C, including the grounded outer terminal:

Q = C V, and W = V^T C V/2.

The complete matrix is symmetric, positive semidefinite and has zero row sums.
Its gauge mode is expected. Dropping the ground terminal produces the usual reduced
positive matrix. Terminal labels are integers; -1 denotes a free dielectric node.

A floating terminal has specified charge, usually zero. The floating-voltage helper
solves the appropriate C_ff block. It must not be modeled by fixing an arbitrary
voltage or removing the conductor. The coupling study archives every unit-voltage
basis and compares matrix energy to the combined field energy.

## Force

At fixed voltage, mechanical force follows the derivative of positive electrical
coenergy. At fixed charge the constraint changes, so the same voltage derivative
cannot be reused with an unchanged sign interpretation.

The Maxwell tensor in a homogeneous electrostatic medium is
T_ij = epsilon(E_i E_j - delta_ij |E|^2/2).
rectangular_force integrates T dot n around a closed grid-aligned contour.
The contour must lie in one homogeneous medium and avoid singular sources.

The maxwell-force study uses the analytic field of a line charge and its grounded
image. Mesh refinement therefore isolates contour quadrature error, rather than
claiming a coupled Poisson-and-force convergence result.

The coax-convergence study retains real nonmonotonic staircase errors. Change
the conductor rasterization, exterior geometry and mesh independently when using
the solver for a new device.
