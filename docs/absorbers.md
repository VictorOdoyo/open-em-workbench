# Absorbing boundaries and convergence

The workbench provides PEC termination, a first-order Mur boundary in one dimension,
and convolutional perfectly matched layers in 1D and scalar 2D TMz.

CPML uses a polynomial loss-rate profile and recursive memory terms on the staggered
electric and magnetic derivatives. The profile has a maximum kappa and a complex
frequency shift alpha. Loss parameters are rates in 1/s; they are not material
conductivities in S/m. The FDFD counterpart uses complex coordinate stretching.

The target_reflection parameter sets a continuum profile design. It is not a
measured numerical reflection guarantee. Finite layer thickness, discretization,
frequency, incidence and proximity to evanescent fields affect the observed result.
A small target cannot fix a poorly resolved layer.

## Reproducible pulse experiment

fdtd-absorber launches the same right-going Gaussian packet into PEC, Mur and CPML
terminations, then measures the peak late probe amplitude relative to the incident
window. A separate sweep uses 8,16,32 and 48 PML cells at the base resolution.
The actual windows, dt, dx, profile and physical thickness are archived.

Changing resolution_scale preserves the physical domain and layer thickness by
scaling node and layer counts. It also changes numerical dispersion. Distinguish
a fixed-grid thickness sweep from a mesh refinement at fixed physical thickness.

## Practical checks

1. Keep sources and analysis probes in the physical interior.
2. Increase PML thickness at fixed interior resolution.
3. Refine the physical grid while preserving physical distances.
4. Move the outer boundary and check whether the observable changes.
5. For pulses, inspect the late time trace and sufficient capture duration.

These implementations do not claim universal PML stability for arbitrary anisotropic,
negative-index, nonlinear or backward-wave media.

Method context: [Meep PML documentation](https://meep.readthedocs.io/en/latest/Perfectly_Matched_Layer/).
