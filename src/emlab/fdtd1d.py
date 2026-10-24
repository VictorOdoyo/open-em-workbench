import numpy as np


class Yee1D:
    """Ey at integer x/time; Hz at half x and the previous half time."""

    def __init__(
        self, nodes, spacing_m, epsilon_r=1.0, conductivity_s_m=0.0, mu_r=1.0, courant=0.95, dt_s=None
    ):
        from .core import MU0, finite_array
        from .timedomain import courant_step, electric_coefficients

        if not isinstance(nodes, int) or isinstance(nodes, bool) or nodes < 5:
            raise ValueError("at least five electric-field nodes required")
        if spacing_m <= 0 or not np.isfinite(spacing_m):
            raise ValueError("positive finite spacing required")
        self.nodes, self.dx = nodes, spacing_m
        self.epsilon = np.broadcast_to(finite_array(epsilon_r, real=True), (nodes,)).copy()
        self.conductivity = np.broadcast_to(finite_array(conductivity_s_m, real=True), (nodes,)).copy()
        self.mu = np.broadcast_to(finite_array(mu_r, real=True), (nodes - 1,)).copy()
        if np.any(self.epsilon <= 0) or np.any(self.mu <= 0) or np.any(self.conductivity < 0):
            raise ValueError("positive constitutive parameters and passive conductivity required")
        limit = courant_step([spacing_m], float(self.epsilon.min()), float(self.mu.min()), 1.0)
        self.dt = limit * courant if dt_s is None else dt_s
        if not 0 < courant <= 1 or not np.isfinite(self.dt) or not 0 < self.dt <= limit * (1 + 1e-12):
            raise ValueError("timestep exceeds the fastest-material Courant limit")
        self.ca, self.cb = electric_coefficients(self.epsilon, self.conductivity, self.dt)
        self.ch = self.dt / (MU0 * self.mu)
        self.e = np.zeros(nodes)
        self.h = np.zeros(nodes - 1)
        self.iteration = 0

    @property
    def electric_time_s(self):
        return self.iteration * self.dt

    @property
    def magnetic_time_s(self):
        return (self.iteration - 0.5) * self.dt

    def _gradient_e(self):
        return np.diff(self.e) / self.dx

    def _gradient_h(self):
        return np.diff(self.h) / self.dx

    def _update_electric(self, derivative, current):
        self.e[1:-1] = self.ca[1:-1] * self.e[1:-1] - self.cb[1:-1] * (derivative + current)

    def step(self, current_density=0):
        from .core import finite_array

        current = np.broadcast_to(finite_array(current_density, real=True), self.e.shape)
        if current[0] != 0 or current[-1] != 0:
            raise ValueError("impressed currents cannot occupy PEC boundary nodes")
        self.h -= self.ch * self._gradient_e()
        self._update_electric(self._gradient_h(), current[1:-1])
        self.e[[0, -1]] = 0
        self.iteration += 1
        return self.e
