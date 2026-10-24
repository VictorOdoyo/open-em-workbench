import numpy as np
import pytest

from emlab import fdtd2d as m


def test_tmz_update_matches_five_point_wave_stencil_and_field_orientations():
    from emlab.grid import Grid2D

    state = m.TMz2D(Grid2D(9, 9, 0.001, 0.001), courant=0.5)
    state.ez[4, 4] = 1
    state.step()
    assert state.ez[4, 4] == pytest.approx(0.5)
    assert state.ez[4, 3] == pytest.approx(0.125)
    assert state.ez[3, 4] == pytest.approx(0.125)
    assert state.hx[3, 4] < 0 and state.hy[4, 3] > 0


def test_embedded_pec_enforces_zero_electric_field_and_rejects_internal_sources():
    from emlab.grid import Grid2D

    grid = Grid2D(9, 9, 0.001, 0.001)
    mask = np.zeros(grid.shape, bool)
    mask[4, 4] = True
    state = m.TMz2D(grid, pec_mask=mask)
    state.ez[4, 4] = 1
    state.step()
    assert np.max(abs(state.ez)) == 0
    with pytest.raises(ValueError, match="PEC"):
        state.step(mask.astype(float))


def test_two_dimensional_pml_absorbs_a_symmetric_outgoing_pulse():
    from emlab.grid import Grid2D

    grid = Grid2D(81, 81, 0.002, 0.002)
    x, y = grid.mesh()
    maxima = []
    for cls, options in ((m.TMz2D, {}), (m.CPMLTMz, {"pml_cells": 12})):
        state = cls(grid, **options)
        state.ez[:] = np.exp(-((x - 0.08) ** 2 + (y - 0.08) ** 2) / 0.01**2)
        state.ez[state.pec] = 0
        for _ in range(350):
            state.step()
        maxima.append(np.linalg.norm(state.ez[15:-15, 15:-15]))
        assert np.all(np.isfinite(state.ez))
        assert np.max(abs(state.ez - state.ez.T)) < 1e-12
    assert maxima[1] < 0.03 * maxima[0]


def test_rectangular_cavity_energy_is_conserved_with_anisotropic_spacing():
    from emlab.grid import from_bounds

    grid = from_bounds((0, 0.2), (0, 0.1), 41, 31)
    state = m.TMz2D(grid)
    x, y = grid.mesh()
    state.ez[:] = np.sin(np.pi * x / 0.2) * np.sin(2 * np.pi * y / 0.1)
    state.ez[state.pec] = 0
    initial = m.stored_energy(state, modified=True)
    for _ in range(300):
        state.step()
    assert m.stored_energy(state, modified=True) == pytest.approx(initial, rel=1e-12)


def test_line_current_integrates_to_requested_amperes_after_refinement():
    from emlab.grid import Grid2D

    for dx in (0.002, 0.001):
        state = m.TMz2D(Grid2D(21, 21, dx, dx))
        current = m.line_current(state, (10, 10), 0.1)
        assert np.sum(current) * state.grid.dx * state.grid.dy == pytest.approx(0.1)


def test_cavity_time_trace_matches_independent_discrete_dispersion_prediction():
    from emlab.core import C0
    from emlab.grid import from_bounds
    from emlab.monitors import ringdown_pole
    from emlab.timedomain import dispersion_frequency

    grid = from_bounds((0, 0.2), (0, 0.1), 41, 31)
    state = m.TMz2D(grid)
    m.cavity_mode(state, 1, 2)
    trace = []
    for _ in range(400):
        state.step()
        trace.append(state.ez[8, 20])
    measured = ringdown_pole(trace, state.dt)
    expected = dispersion_frequency([np.pi / 0.2, 2 * np.pi / 0.1], [grid.dx, grid.dy], state.dt, C0) / (
        2 * np.pi
    )
    assert measured["frequency_hz"] == pytest.approx(expected, rel=1e-10)
