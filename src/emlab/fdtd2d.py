import numpy as np


class TMz2D:
    """Ez nodes, Hx y-half edges, Hy x-half edges; x is the final array axis."""

    def __init__(
        self, grid, epsilon_r=1.0, conductivity_s_m=0.0, mu_r=1.0, courant=0.95, dt_s=None, pec_mask=None
    ):
        from .core import MU0, finite_array
        from .grid import boundary_mask
        from .timedomain import courant_step, electric_coefficients

        self.grid = grid
        self.epsilon = np.broadcast_to(finite_array(epsilon_r, real=True), grid.shape).copy()
        self.conductivity = np.broadcast_to(finite_array(conductivity_s_m, real=True), grid.shape).copy()
        mu = finite_array(mu_r, real=True)
        if mu.ndim != 0 or mu <= 0 or np.any(self.epsilon <= 0) or np.any(self.conductivity < 0):
            raise ValueError("positive scalar mu and passive scalar dielectric map required")
        self.mu = float(mu)
        limit = courant_step([grid.dx, grid.dy], float(self.epsilon.min()), self.mu, 1.0)
        self.dt = limit * courant if dt_s is None else dt_s
        if not 0 < courant <= 1 or not np.isfinite(self.dt) or not 0 < self.dt <= limit * (1 + 1e-12):
            raise ValueError("timestep exceeds the two-dimensional Courant limit")
        self.ca, self.cb = electric_coefficients(self.epsilon, self.conductivity, self.dt)
        self.ch = self.dt / (MU0 * self.mu)
        self.pec = boundary_mask(grid)
        if pec_mask is not None:
            mask = np.asarray(pec_mask)
            if mask.shape != grid.shape or mask.dtype != bool:
                raise ValueError("boolean embedded PEC mask matching grid required")
            self.pec |= mask
        self.ez = np.zeros(grid.shape)
        self.hx = np.zeros((grid.ny - 1, grid.nx))
        self.hy = np.zeros((grid.ny, grid.nx - 1))
        self.iteration = 0

    def _gradients_e(self):
        return np.diff(self.ez, axis=0) / self.grid.dy, np.diff(self.ez, axis=1) / self.grid.dx

    def _curl_h(self):
        dx = np.diff(self.hy, axis=1)[1:-1, :] / self.grid.dx
        dy = np.diff(self.hx, axis=0)[:, 1:-1] / self.grid.dy
        return dx - dy

    def step(self, current_density=0):
        from .core import finite_array

        current = np.broadcast_to(finite_array(current_density, real=True), self.grid.shape)
        if np.any(current[self.pec] != 0):
            raise ValueError("impressed current cannot occupy PEC nodes")
        self.ez[self.pec] = 0
        derivative_y, derivative_x = self._gradients_e()
        self.hx -= self.ch * derivative_y
        self.hy += self.ch * derivative_x
        self.ez[1:-1, 1:-1] = self.ca[1:-1, 1:-1] * self.ez[1:-1, 1:-1] + self.cb[1:-1, 1:-1] * (
            self._curl_h() - current[1:-1, 1:-1]
        )
        self.ez[self.pec] = 0
        self.iteration += 1
        return self.ez


class CPMLTMz(TMz2D):
    def __init__(
        self, grid, pml_cells=16, target_reflection=1e-8, kappa_max=3.0, alpha_max_s=None, **options
    ):
        from .core import C0
        from .timedomain import pml_coefficients, pml_profile

        super().__init__(grid, **options)
        if (
            not isinstance(pml_cells, int)
            or isinstance(pml_cells, bool)
            or not 2 <= pml_cells < (min(grid.nx, grid.ny) - 1) / 2
        ):
            raise ValueError("nonoverlapping PML with at least two cells per side required")
        self.pml_cells = pml_cells
        alpha_max_s = 0.05 * C0 / min(grid.dx, grid.dy) if alpha_max_s is None else alpha_max_s

        def axis(nodes, spacing):
            extent = (nodes - 1) * spacing
            electric = np.arange(nodes) * spacing
            magnetic = (np.arange(nodes - 1) + 0.5) * spacing
            result = []
            for coordinate in (electric, magnetic):
                profile = pml_profile(
                    coordinate,
                    0,
                    extent,
                    pml_cells * spacing,
                    target_reflection,
                    kappa_max=kappa_max,
                    alpha_max_s=alpha_max_s,
                )
                result.append(pml_coefficients(*profile, self.dt))
            return result

        self.xe, self.xh = axis(grid.nx, grid.dx)
        self.ye, self.yh = axis(grid.ny, grid.dy)
        self.psi_hx = np.zeros_like(self.hx)
        self.psi_hy = np.zeros_like(self.hy)
        self.psi_ex = np.zeros((grid.ny - 2, grid.nx - 2))
        self.psi_ey = np.zeros_like(self.psi_ex)

    def _gradients_e(self):
        dy, dx = super()._gradients_e()
        by, cy, ky = [value[:, None] for value in self.yh]
        bx, cx, kx = [value[None, :] for value in self.xh]
        self.psi_hx = by * self.psi_hx + cy * dy
        self.psi_hy = bx * self.psi_hy + cx * dx
        return dy / ky + self.psi_hx, dx / kx + self.psi_hy

    def _curl_h(self):
        dx = np.diff(self.hy, axis=1)[1:-1, :] / self.grid.dx
        dy = np.diff(self.hx, axis=0)[:, 1:-1] / self.grid.dy
        bx, cx, kx = [value[None, 1:-1] for value in self.xe]
        by, cy, ky = [value[1:-1, None] for value in self.ye]
        self.psi_ex = bx * self.psi_ex + cx * dx
        self.psi_ey = by * self.psi_ey + cy * dy
        return dx / kx + self.psi_ex - dy / ky - self.psi_ey


def stored_energy(state, modified=False):
    from .core import EPS0, MU0

    area = state.grid.dx * state.grid.dy
    electric = 0.5 * EPS0 * np.sum(state.epsilon * state.ez**2) * area
    if modified:
        if type(state) is not TMz2D or np.any(state.conductivity != 0):
            raise ValueError("modified invariant requires lossless nondispersive PEC TMz2D")
        dy, dx = state._gradients_e()
        next_hx, next_hy = state.hx - state.ch * dy, state.hy + state.ch * dx
        magnetic = 0.5 * MU0 * state.mu * (np.sum(state.hx * next_hx) + np.sum(state.hy * next_hy)) * area
    else:
        magnetic = 0.5 * MU0 * state.mu * (np.sum(state.hx**2) + np.sum(state.hy**2)) * area
    return float(electric + magnetic)


def line_current(state, index_yx, amplitude_a):
    from .core import finite_array

    amplitude = finite_array(amplitude_a, real=True)
    if (
        len(index_yx) != 2
        or any(not isinstance(i, int) for i in index_yx)
        or not 0 <= index_yx[0] < state.grid.ny
        or not 0 <= index_yx[1] < state.grid.nx
        or amplitude.ndim != 0
    ):
        raise ValueError("valid electric node index and scalar line current required")
    iy, ix = index_yx
    margin = getattr(state, "pml_cells", 0) + 1
    if (
        state.pec[iy, ix]
        or not margin <= iy < state.grid.ny - margin
        or not margin <= ix < state.grid.nx - margin
    ):
        raise ValueError("line current must lie outside PEC and PML regions")
    result = np.zeros(state.grid.shape)
    result[iy, ix] = amplitude / (state.grid.dx * state.grid.dy)
    return result
