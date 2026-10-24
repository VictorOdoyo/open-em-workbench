# Dispersive materials

Under exp(+i omega t), the single-pole Debye model is

epsilon(omega) = epsilon_inf + (epsilon_static-epsilon_inf)/(1+i omega tau).

Passivity requires epsilon_static >= epsilon_inf > 0 and tau > 0.
Its imaginary part is negative at positive frequency. Lorentz and Drude frequency
models follow this same convention; their resonant limits and passive branches
are checked independently.

## Time-domain auxiliary polarization

Debye1D stores P and solves tau dP/dt + P = EPS0 delta_epsilon E with a trapezoidal
recurrence. The E update includes the new P implicitly, along with conductive loss.
It is not valid to insert a frequency-dependent complex epsilon directly into a
real-valued time-step coefficient.

The finite-dt susceptibility differs slightly from the continuum formula. The code
provides both responses. The fdtd-debye study records the discrete/continuum
difference and compares the fully integrated slab scattering against the continuum
multilayer solution. Final polarization is saved in C/m^2.

At interfaces, the electric nodes use the stated half-cell arithmetic material
averages. That spatial averaging and ADE time error are separate approximations.

## Energy and restrictions

For a passive Debye strength, polarization energy density is P^2/(2 EPS0 delta_epsilon).
Zero-strength cells contribute zero polarization energy. This storage term is
separate from irreversible relaxation loss; it does not imply that a lossy state
preserves the lossless leapfrog invariant.

Only the Debye model has a time-domain ADE implementation here. Frequency-domain
Lorentz/Drude helpers do not imply corresponding time-domain oscillators.

[Meep material documentation](https://meep.readthedocs.io/en/latest/Materials/) describes
related auxiliary-polarization methods. Meep uses the opposite phasor sign; copy
neither its imaginary signs nor its normalization into this package without conversion.
