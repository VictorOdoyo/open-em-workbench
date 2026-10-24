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


def test_spatial_magnetic_collocation_recovers_affine_fields_at_interior_nodes():
    from emlab.grid import Grid2D

    grid = Grid2D(9, 7, 0.001, 0.002)
    state = m.TMz2D(grid)
    state.hx[:] = ((np.arange(grid.ny - 1) + 0.5) * grid.dy)[:, None]
    state.hy[:] = ((np.arange(grid.nx - 1) + 0.5) * grid.dx)[None, :]
    hx, hy = m.magnetic_nodes(state)
    assert np.allclose(hx[1:-1, 1:-1], grid.mesh()[1][1:-1, 1:-1])
    assert np.allclose(hy[1:-1, 1:-1], grid.mesh()[0][1:-1, 1:-1])


def test_two_dimensional_recording_has_deterministic_capture_times_and_probe_axes():
    from emlab.grid import Grid2D
    from emlab.sources import ricker

    state = m.CPMLTMz(Grid2D(41, 35, 0.002, 0.002), pml_cells=6)

    def current(time):
        return m.line_current(state, (17, 20), 0.001 * ricker(time, 0.1e-9, 5e9))

    result = m.evolve(state, 83, current, [(17, 15), (17, 25)], 20)
    assert result["electric"].shape == (83, 2)
    assert np.allclose(result["electric"][:, 0], result["electric"][:, 1], atol=1e-12)
    assert np.array_equal(result["snapshot_steps"], [0, 20, 40, 60, 80, 83])
    assert result["snapshots"].shape == (6, 35, 41)
    assert np.all(np.isfinite(result["energy_j_m"]))


def test_measured_lossy_cavity_pole_matches_independent_discrete_recurrence():
    from emlab.grid import from_bounds
    from emlab.monitors import ringdown_pole

    grid = from_bounds((0, 0.2), (0, 0.1), 41, 31)
    state = m.TMz2D(grid, epsilon_r=2, conductivity_s_m=0.01)
    m.cavity_mode(state, 1, 1)
    values = []
    for _ in range(500):
        state.step()
        values.append(state.ez[15, 20])
    measured = ringdown_pole(values, state.dt)
    roots = m.conductive_cavity_pole(grid, 1, 1, state.dt, 2, 0.01)
    pole = roots[roots.imag > 0][0]
    expected_frequency = np.angle(pole) / (2 * np.pi * state.dt)
    expected_decay = -np.log(abs(pole)) / state.dt
    assert measured["frequency_hz"] == pytest.approx(expected_frequency, rel=1e-10)
    assert measured["decay_rate_s"] == pytest.approx(expected_decay, rel=1e-10)


def test_isolated_cavity_initializer_rejects_internal_pec_and_nonuniform_damping():
    from emlab.grid import Grid2D

    grid = Grid2D(9, 9, 0.001, 0.001)
    pec = np.zeros(grid.shape, bool)
    pec[4, 4] = True
    sigma = np.zeros(grid.shape)
    sigma[3, 3] = 0.1
    for state in (m.TMz2D(grid, pec_mask=pec), m.TMz2D(grid, conductivity_s_m=sigma)):
        with pytest.raises(ValueError, match="uniform rectangular"):
            m.cavity_mode(state)
