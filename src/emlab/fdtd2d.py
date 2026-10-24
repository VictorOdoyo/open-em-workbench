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
