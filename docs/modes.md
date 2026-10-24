# Eigenmodes and guided waves

For a uniform rectangular hollow guide, cutoff is
c/(2 sqrt(epsilon_r mu_r)) sqrt((m/a)^2+(n/b)^2).
TE permits one zero index but excludes TE00. TM requires both indices positive.

The two-dimensional scalar cavity problems are generalized symmetric eigenproblems:

- TMz: K(1/mu_r) Ez = k0^2 M(epsilon_r) Ez, with zero boundary Ez.
- TEz: K(1/epsilon_r) Hz = k0^2 M(mu_r) Hz, with natural Neumann boundary.

The Neumann constant field is removed before reporting physical TE modes.
These scalar reductions do not represent general hybrid modes in longitudinally
varying, anisotropic or arbitrary vector waveguides.

Mass normalization is weighted by the relevant material and dual volume.
A mode's sign is arbitrary. Within a degenerate cluster, eigenvectors may rotate
while spanning the same physical subspace. Gram and projection helpers test the
subspace, rather than requiring unstable pointwise vector equality.

## Symmetric dielectric slab

The analytical TE slab solver brackets even/odd transcendental roots in their
physical intervals. A separate finite-difference transverse eigenproblem yields
effective index and field profiles. Guided modes satisfy n_clad < neff < n_core.

A finite computational cladding replaces an infinite decay region. Refining dx
at fixed cladding thickness cannot remove this truncation error. The slab-guidance
study states its domain and plots the core and evanescent tails. Mode amplitudes
are normalized for comparison, not presented as one-watt vector power modes.

## Perturbations

For a small dielectric perturbation, first-order frequency shift is proportional
to -integral(delta_epsilon |E|^2)/(2 integral(epsilon |E|^2)).
The mode-perturbation study re-solves every material increment and shows where that
linear prediction departs from the actual discrete eigenvalue.

For implementation details see
[SciPy eigsh](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html)
and the [MIT waveguide lecture](https://www.ocw.mit.edu/courses/6-013-electromagnetics-and-applications-spring-2009/d9425aa2b1acd0c1fc121965ad7eafec_MIT6_013S09_lec16.pdf).
