# Sensitivity and material design

The fixed-voltage energy is W = EPS0 phi^T K(epsilon) phi/2.
With no free charge and fixed conductor potentials, its derivative can be evaluated
at the stationary solved field without differentiating phi explicitly.

For an edge joining materials a and b, the harmonic mean is 2ab/(a+b).
Its derivatives are 2b^2/(a+b)^2 and 2a^2/(a+b)^2. energy_sensitivity
accumulates these edge contributions at nodes, including edge geometry and voltage
differences. The result is dW/d(epsilon_r at node), in J/m.

The sensitivity study perturbs individual material nodes in both directions and
independently re-solves the fields. Comparing these energies to the analytic
discrete gradient checks the complete operator derivative, not a frozen-field
difference that omits the actual solve.

## Material budget

optimize_dielectric maximizes stored energy at fixed voltage subject to positive
pointwise permittivity bounds and a dual-volume weighted mean.
A clipped scalar shift projects each trial onto the material budget. Backtracking
accepts an energy increase satisfying an ascent condition.

The final capacitance is 2W/V^2 for the single driven terminal. The report includes
every accepted objective, step, final dielectric map, potential and budget defect.

This is a local optimization in a relaxed nodal material space. It has no minimum
feature size, binary-material filter, fabrication process, global-optimum guarantee
or full-wave objective. The material budget is a mathematical mean permittivity,
not an assumed physical mixture law for arbitrary composites.

A new objective or free-charge constraint needs its own derivative derivation;
do not reuse the fixed-voltage no-free-charge formula without checking stationarity.
