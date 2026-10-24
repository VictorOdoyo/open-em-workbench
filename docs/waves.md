# Waves, multilayers and microwave networks

waves defines passive propagation and impedance branches, TE/TM Fresnel coefficients,
critical and Brewster angles, peak-phasor Poynting flux and forward/backward separation.
A propagating external port has real positive power admittance. Evanescent internal
layers may carry reactive fields without defining an independent propagating power port.

## Network normalization

networks converts ABCD matrices to power-wave S parameters with distinct positive real
reference impedances on the two ports. Reciprocity in power normalization means S12=S21.
Passivity requires the largest singular value of S to be at most one; |S21| alone
does not test an arbitrary two-port excitation.

Redheffer star products cascade scattering networks without introducing exponentially
growing internal transfer matrices. Reference-impedance renormalization uses wave
relations directly, avoiding a singular impedance matrix for ideal through networks.
Group delay is -d(unwrapped phase)/d omega. A sparse or poorly illuminated phase sweep
does not support a reliable derivative.

## Dielectric layers

The raw layer network uses tangential electric amplitudes and passive longitudinal
wavenumbers. Interface continuity and decaying propagation are combined stably;
external amplitudes are then converted to flux-normalized reflection/transmission.

Internal forward and backward fields provide layer boundary flux differences.
The absorbing-stack study independently integrates volume dissipation and compares
it with these flux losses and global 1-R-T. Its constant complex permittivities
are finite-band models, not causal broadband time-domain materials.

The coating study uses the quarter-wave index and thickness at a design frequency.
The Bragg study shows a finite mirror, its internal standing field, and the Bloch
attenuation of the corresponding periodic cell. Finite-stack transmission and
infinite-periodic Bloch behavior are different reported quantities.

## RLGC lines

transmission evaluates gamma=sqrt((R+i omega L)(G+i omega C)) with passive branches.
The stable line S formulation handles large attenuation; transfer matrices reject
overflow-prone cases. Open and short loads are explicit limits.

Coaxial skin resistance assumes a good conductor and skin depth small compared with
conductor radii. Dielectric shunt loss uses the specified loss tangent. The example
matching section is a separate ideal quarter-wave transformer, not a claim that a
lossy broadband coax can be perfectly matched by one fixed section.
