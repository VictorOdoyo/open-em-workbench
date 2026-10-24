# Conservative Cartesian operators

The electrostatic discretization is a weighted graph over Cartesian nodes.
Each oriented edge has incidence row (-1,+1), length l, and orthogonal dual-face
measure a. The stiffness matrix is K = B^T diag(epsilon_edge a/l) B.

Material values are positive nodal relative permittivities. Each edge uses
the harmonic mean 2 epsilon_i epsilon_j/(epsilon_i+epsilon_j). This represents
two half-edge dielectric resistances in series and preserves normal displacement
through a grid-aligned material interface.

Dual control volumes have full interior area, half area on outer edges and
quarter area at corners. The consistent charge load is rho times dual area / EPS0.
The divergence is the negative adjoint of the edge gradient with these measures.

## Dirichlet conditions

Constrained potentials are eliminated symmetrically. The free-node problem is
K_ff phi_f = load_f - K_fc phi_c. Sparse direct solution is the default;
conjugate gradients with a diagonal preconditioner are available for positive
definite reduced systems. Both routes report an independently recomputed residual.

A boundary with no constrained nodes and no grounding leaves a gauge nullspace.
Use physical conductor labels or an explicit fixed-potential mask. Leaving an
edge unconstrained gives the natural zero-flux boundary of this variational operator.

## Derived quantities

Energy is EPS0 phi^T K phi/2. Conductor charges are the constrained-node reactions,
after subtracting distributed source charge. Consequently charge, energy and source
work use one operator rather than unrelated pointwise formulas.

Rasterizing a circle changes the discrete geometry when the mesh changes.
The centered bulk stencil's manufactured-solution order does not imply second-order
capacitance convergence for staircase conductors. Both behaviors are shown separately.

The scalar vector-potential solver reuses this conservative structure with inverse
permeability weighting; frequency-domain waves add a complex mass term.
