import numpy as np

from emlab import fdfd as m


def test_frequency_domain_pml_has_negative_imaginary_stretch_and_real_interior():
    from emlab.grid import Grid2D

    grid = Grid2D(51, 41, 0.005, 0.005)
    stretch = m.coordinate_stretch(grid, 1e9, 8)
    assert stretch["xe"].shape == (51,) and stretch["xh"].shape == (50,)
    assert stretch["xe"][0].imag < 0
    assert stretch["xe"][25] == 1
    assert np.allclose(stretch["ye"], stretch["ye"][::-1])


def test_closed_helmholtz_operator_matches_analytic_discrete_sine_eigenvalue():
    from emlab.core import C0
    from emlab.grid import from_bounds

    grid = from_bounds((0, 0.2), (0, 0.1), 31, 21)
    x, y = grid.mesh()
    field = np.sin(np.pi * x / 0.2) * np.sin(np.pi * y / 0.1)
    model = m.operator(grid, 0.5e9, 2, pml_cells=0)
    value = (model["matrix"] @ field.ravel()).reshape(grid.shape)
    eigenvalue = (
        4 * np.sin(np.pi / (2 * (grid.nx - 1))) ** 2 / grid.dx**2
        + 4 * np.sin(np.pi / (2 * (grid.ny - 1))) ** 2 / grid.dy**2
    )
    expected = (eigenvalue - 2 * (2 * np.pi * 0.5e9 / C0) ** 2) * field * model["coordinate_volume"]
    assert np.allclose(value[1:-1, 1:-1], expected[1:-1, 1:-1], rtol=1e-11, atol=1e-13)
    assert np.max(abs((model["matrix"] - model["matrix"].T).data), initial=0) < 1e-12


def test_current_driven_helmholtz_matches_manufactured_cavity_solution():
    from emlab.core import C0, MU0
    from emlab.grid import from_bounds

    frequency = 0.5e9
    grid = from_bounds((0, 0.2), (0, 0.1), 31, 21)
    x, y = grid.mesh()
    field = np.sin(np.pi * x / 0.2) * np.sin(np.pi * y / 0.1)
    eigenvalue = (
        4 * np.sin(np.pi / (2 * (grid.nx - 1))) ** 2 / grid.dx**2
        + 4 * np.sin(np.pi / (2 * (grid.ny - 1))) ** 2 / grid.dy**2
    )
    current = 1j * (eigenvalue - (2 * np.pi * frequency / C0) ** 2) / (2 * np.pi * frequency * MU0) * field
    current[[0, -1], :] = 0
    current[:, [0, -1]] = 0
    result = m.solve(grid, frequency, current, pml_cells=0)
    assert np.max(abs(result["electric"] - field)) < 1e-12
    assert result["residual"] < 1e-12


def test_open_domain_current_field_matches_outgoing_hankel_green_function():
    from emlab.core import MU0, relative_error
    from emlab.grid import from_bounds

    grid = from_bounds((-0.2, 0.2), (-0.2, 0.2), 101, 101)
    current = np.zeros(grid.shape)
    current[50, 50] = 0.001 / (grid.dx * grid.dy)
    field = m.solve(grid, 1e9, current, pml_cells=14)["electric"]
    locations = np.array([60, 65, 70, 75, 80])
    radius = grid.x[locations]
    expected = -2j * np.pi * 1e9 * MU0 * 0.001 * m.green_2d(1e9, radius)
    assert relative_error(expected, field[50, locations]) < 0.015


def test_oblique_grid_plane_wave_satisfies_interior_helmholtz_equation():
    from emlab.grid import from_bounds

    grid = from_bounds((-0.1, 0.1), (-0.1, 0.1), 31, 27)
    incident, _ = m.plane_wave(grid, 1e9, angle_radians=0.7)
    model = m.operator(grid, 1e9, pml_cells=0)
    residual = (model["matrix"] @ incident.ravel()).reshape(grid.shape)
    assert np.max(abs(residual[1:-1, 1:-1])) < 1e-12


def test_frequency_domain_magnetic_field_has_forward_poynting_orientation():
    from emlab.core import ETA0
    from emlab.grid import from_bounds

    grid = from_bounds((-0.1, 0.1), (-0.1, 0.1), 41, 31)
    electric, wavevector = m.plane_wave(grid, 1e9)
    hx, hy = m.magnetic_fields(grid, 1e9, electric, pml_cells=0)
    midpoint_x = (grid.x[:-1] + grid.x[1:]) / 2
    expected = -np.exp(-1j * wavevector[0] * midpoint_x) / ETA0
    assert np.max(abs(hx)) < 1e-15
    assert np.allclose(hy[15], expected, atol=1e-14)
