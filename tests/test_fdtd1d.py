import numpy as np
import pytest

from emlab import fdtd1d as m


def test_yee_state_has_explicit_space_time_staggering_and_courant_guard():
    from emlab.core import C0

    state = m.Yee1D(101, 0.001)
    assert state.e.shape == (101,) and state.h.shape == (100,)
    assert state.magnetic_time_s == -state.dt / 2
    with pytest.raises(ValueError, match="Courant"):
        m.Yee1D(101, 0.001, dt_s=0.002 / C0)
    fast = m.Yee1D(101, 0.001, epsilon_r=0.25)
    assert fast.dt == pytest.approx(state.dt / 2)


def test_yee_impulse_update_matches_independent_three_point_wave_stencil():
    state = m.Yee1D(7, 0.001, courant=0.5)
    state.e[3] = 1
    state.step()
    assert np.allclose(state.e, [0, 0, 0.25, 0.5, 0.25, 0, 0])
    assert state.iteration == 1


def test_impressed_current_has_ampere_sign_and_si_scaling():
    from emlab.core import EPS0

    state = m.Yee1D(7, 0.001)
    current = np.zeros(7)
    current[3] = 0.1
    state.step(current)
    assert state.e[3] == pytest.approx(-state.dt / EPS0 * 0.1)
    assert state.electric_time_s - state.magnetic_time_s == pytest.approx(state.dt / 2)


def test_cpml_reduces_a_returning_pulse_compared_with_pec():
    from emlab.core import C0, ETA0

    traces = []
    for cls, options in ((m.Yee1D, {}), (m.CPML1D, {"pml_cells": 40})):
        state = cls(401, 0.001, **options)
        state.e[:] = np.exp(-(((np.arange(401) * state.dx - 0.2) / 0.015) ** 2))
        positions = (np.arange(400) + 0.5) * state.dx + C0 * state.dt / 2
        state.h[:] = np.exp(-(((positions - 0.2) / 0.015) ** 2)) / ETA0
        trace = []
        for _ in range(500):
            state.step()
            trace.append(state.e[250])
        traces.append(np.array(trace))
    time = np.arange(500) * state.dt
    late = (time > 0.65e-9) & (time < 1.35e-9)
    assert np.max(abs(traces[0][late])) > 0.8
    assert np.max(abs(traces[1][late])) < 0.002


def test_unit_courant_mur_boundary_transmits_a_discrete_impulse_without_reflection():
    from emlab.core import ETA0

    state = m.Mur1D(101, 0.001, courant=1.0)
    state.e[30] = 1
    state.h[29] = 1 / ETA0
    for _ in range(250):
        state.step()
    assert np.max(abs(state.e)) < 1e-12


def test_lossless_pec_leapfrog_energy_is_conserved_at_machine_precision():
    state = m.Yee1D(121, 0.001, courant=0.9)
    state.e[:] = np.sin(np.pi * np.arange(121) / 120)
    initial = m.stored_energy(state, modified=True)
    energies = []
    for _ in range(500):
        state.step()
        energies.append(m.stored_energy(state, modified=True))
    assert np.max(abs(np.array(energies) - initial)) / initial < 1e-12
    assert m.stored_energy(state) > 0


def test_smooth_packet_moves_at_the_material_wave_speed():
    from emlab.core import C0

    state = m.Yee1D(501, 0.001, epsilon_r=4)
    m.gaussian_packet(state, 0.15, 0.02)
    for _ in range(80):
        state.step()
    measured = np.argmax(state.e) * state.dx
    assert measured == pytest.approx(0.15 + C0 / 2 * 80 * state.dt, abs=0.0015)


def test_sheet_current_integral_is_independent_of_grid_spacing():
    for dx in (0.001, 0.0005):
        state = m.Yee1D(101, dx)
        source = m.sheet_current(state, 50, 0.3)
        assert np.sum(source) * dx == pytest.approx(0.3)
    with pytest.raises(ValueError):
        m.sheet_current(m.CPML1D(101, 0.001, pml_cells=12), 5, 1)
