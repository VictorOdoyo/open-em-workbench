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


class CPML1D(Yee1D):
    def __init__(
        self,
        nodes,
        spacing_m,
        pml_cells=32,
        target_reflection=1e-8,
        kappa_max=3.0,
        alpha_max_s=None,
        **options,
    ):
        from .core import C0
        from .timedomain import pml_coefficients, pml_profile

        super().__init__(nodes, spacing_m, **options)
        if (
            not isinstance(pml_cells, int)
            or isinstance(pml_cells, bool)
            or not 2 <= pml_cells < (nodes - 1) / 2
        ):
            raise ValueError("at least two nonoverlapping PML cells on each side required")
        self.pml_cells = pml_cells
        alpha_max_s = 0.05 * C0 / spacing_m if alpha_max_s is None else alpha_max_s
        extent = (nodes - 1) * spacing_m
        x_e = np.arange(nodes) * spacing_m
        x_h = (np.arange(nodes - 1) + 0.5) * spacing_m
        e_profile = pml_profile(
            x_e,
            0,
            extent,
            pml_cells * spacing_m,
            target_reflection,
            kappa_max=kappa_max,
            alpha_max_s=alpha_max_s,
        )
        h_profile = pml_profile(
            x_h,
            0,
            extent,
            pml_cells * spacing_m,
            target_reflection,
            kappa_max=kappa_max,
            alpha_max_s=alpha_max_s,
        )
        self.be, self.ce, self.ke = [v[1:-1] for v in pml_coefficients(*e_profile, self.dt)]
        self.bh, self.cph, self.kh = pml_coefficients(*h_profile, self.dt)
        self.psi_e = np.zeros(nodes - 2)
        self.psi_h = np.zeros(nodes - 1)

    def _gradient_e(self):
        raw = super()._gradient_e()
        self.psi_h = self.bh * self.psi_h + self.cph * raw
        return raw / self.kh + self.psi_h

    def _gradient_h(self):
        raw = super()._gradient_h()
        self.psi_e = self.be * self.psi_e + self.ce * raw
        return raw / self.ke + self.psi_e


class Mur1D(Yee1D):
    def step(self, current_density=0):
        from .core import C0

        old = self.e[[0, 1, -2, -1]].copy()
        super().step(current_density)
        left_speed = C0 / np.sqrt(self.epsilon[0] * self.mu[0])
        right_speed = C0 / np.sqrt(self.epsilon[-1] * self.mu[-1])
        q_left = (left_speed * self.dt - self.dx) / (left_speed * self.dt + self.dx)
        q_right = (right_speed * self.dt - self.dx) / (right_speed * self.dt + self.dx)
        self.e[0] = old[1] + q_left * (self.e[1] - old[0])
        self.e[-1] = old[2] + q_right * (self.e[-2] - old[3])
        return self.e


def stored_energy(state, modified=False):
    from .core import EPS0, MU0

    electric = 0.5 * EPS0 * np.sum(state.epsilon * state.e**2) * state.dx
    magnetic = 0.5 * MU0 * np.sum(state.mu * state.h**2) * state.dx
    if modified:
        if type(state) is not Yee1D or np.any(state.conductivity != 0):
            raise ValueError("modified invariant applies only to nondispersive lossless PEC Yee1D")
        future_h = state.h - state.ch * np.diff(state.e) / state.dx
        magnetic = 0.5 * MU0 * np.sum(state.mu * state.h * future_h) * state.dx
    return float(electric + magnetic)


def gaussian_packet(state, center_m, width_m, amplitude_v_m=1.0):
    from .core import C0, ETA0

    if width_m <= 0 or not 0 < center_m < (state.nodes - 1) * state.dx:
        raise ValueError("positive pulse width and interior center required")
    if np.ptp(state.epsilon) > 0 or np.ptp(state.mu) > 0 or np.any(state.conductivity != 0):
        raise ValueError("initial traveling packet requires a homogeneous lossless medium")
    speed = C0 / np.sqrt(state.epsilon[0] * state.mu[0])
    eta = ETA0 * np.sqrt(state.mu[0] / state.epsilon[0])
    electric_x = np.arange(state.nodes) * state.dx
    magnetic_x = (np.arange(state.nodes - 1) + 0.5) * state.dx + speed * state.dt / 2
    state.e[:] = amplitude_v_m * np.exp(-(((electric_x - center_m) / width_m) ** 2))
    state.h[:] = amplitude_v_m / eta * np.exp(-(((magnetic_x - center_m) / width_m) ** 2))
    state.e[[0, -1]] = 0
