# Primary references and attribution

Implementations in src/emlab are written for this project. NumPy, SciPy and
Matplotlib supply numerical arrays, sparse solvers, special functions and plotting.
The following primary documentation provides constants and method context;
no external solver output is used as a substitute for the implemented updates.

| Topic | Primary source | Use here |
| --- | --- | --- |
| Constants | [NIST 2022 constants](https://physics.nist.gov/cuu/pdf/wall_2022.pdf) | MU0 central value, exact c, derived EPS0 |
| Magnetic constant | [NIST MU0](https://physics.nist.gov/cgi-bin/cuu/Value?mu0%7Csearch_for=magnetic) | Explicit post-2019 SI convention |
| Absorbing boundaries | [Meep PML](https://meep.readthedocs.io/en/latest/Perfectly_Matched_Layer/) | Continuum versus finite-grid behavior, convergence |
| Dispersive materials | [Meep materials](https://meep.readthedocs.io/en/latest/Materials/) | Auxiliary polarization and passive model interpretation |
| Modal normalization | [Meep mode decomposition](https://meep.readthedocs.io/en/latest/Mode_Decomposition/) | Propagating power and subspace interpretation |
| Reference subtraction | [Meep mode tutorial](https://meep.readthedocs.io/en/latest/Python_Tutorials/Mode_Decomposition/) | Separate incident-reference normalization context |
| Sparse eigenproblems | [SciPy eigsh](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html) | Generalized scalar eigenproblem implementation |
| Outgoing cylindrical waves | [SciPy Hankel2](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.hankel2.html) | Outgoing exp(+i omega t) special function |
| Rectangular waveguides | [MIT 6.013 lecture 16](https://www.ocw.mit.edu/courses/6-013-electromagnetics-and-applications-spring-2009/d9425aa2b1acd0c1fc121965ad7eafec_MIT6_013S09_lec16.pdf) | TE/TM cutoff and propagation interpretation |

Meep uses exp(-i omega t), opposite to this package. A material's imaginary part
and outgoing Hankel branch must be converted when comparing equations.

Analytical references in this repository include manufactured Poisson fields,
coax and parallel-plate capacitance, image charges, Fresnel interfaces, rectangular
cutoffs, symmetric slab roots, loop elliptic integrals and cylindrical harmonic
scattering. Agreement with one reference tests that regime; it does not validate
all geometries, materials or frequencies supported by a function's signature.
