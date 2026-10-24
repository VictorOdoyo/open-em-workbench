# Units and signs

Coordinates are metres, frequency is hertz, time is seconds, conductivity is S/m,
electric field is V/m, magnetic field is A/m, magnetic flux density is tesla,
charge density is C/m^3, and impressed volume current is A/m^2.
Relative permittivity and permeability are dimensionless.

The exact speed of light is 299792458 m/s. The magnetic constant uses the NIST 2022
CODATA central value 1.25663706127e-6 H/m. EPS0 is derived from 1/(MU0 C0^2);
MU0 is not treated as an exact post-2019 SI constant.

## Frequency domain

The real field is Re{E exp(+i omega t)}. A forward field is E+ exp(-gamma z),
where gamma = alpha + i beta. Outgoing cylindrical radiation uses Hankel functions
of the second kind. Conductivity contributes -i sigma/(omega EPS0) to relative
permittivity. Passive Debye, Lorentz and Drude materials follow the same sign.

Field amplitudes are peak phasors. Average Poynting flux is
Re{E cross conjugate(H)}/2 and Joule heating is sigma |E|^2/2.
A Fresnel electric-field amplitude is not generally a power-wave S parameter.
External propagating ports are flux-normalized before reporting R and T.

## Spatial arrays

A Grid2D field has shape (ny,nx). The final index changes x.
Finite-volume E components live on their corresponding faces. Nodal interpolations
are convenient for maps; conservative flux, charge and energy use discrete edges.

Two-dimensional electrostatic capacitance is F/m, energy is J/m and force is N/m,
all per unit out-of-plane length. The 2D TMz line current is A; its one-cell
volume current density is I/(dx dy). A 1D sheet current is A/m and is divided by dx.

## Time domain

Yee1D stores Ey at x nodes at integer time and Hz at half-x points half a timestep
earlier. TMz2D stores Ez at nodes, Hx on half-y edges and Hy on half-x edges.
A trace always includes separate electric_time_s and magnetic_time_s arrays.
Spatial collocation of H alone does not remove the temporal half-step.

Never combine unshifted E/H FFT phases or compare their samples as simultaneous
fields. Fourier integration uses exp(-i omega t) and the recorded time origin,
which recovers the workbench's positive-time phasor.

See the [primary references](references.md) for source constants and method context.
