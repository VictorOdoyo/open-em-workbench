import numpy as np
import pytest

from emlab import magnetostatics as m


def test_finite_wire_field_matches_closed_form_and_right_hand_rule():
    from emlab.core import MU0

    field = m.segment_field([[1, 0, 0], [0, 0, 3]], [0, 0, -1], [0, 0, 1], 2)
    assert field[0, 1] == pytest.approx(MU0 * 2 / (4 * np.pi) * np.sqrt(2))
    assert np.allclose(field[0, [0, 2]], 0)
    assert np.allclose(field[1], 0)
    with pytest.raises(ValueError, match="singular"):
        m.segment_field([0, 0, 0], [0, 0, -1], [0, 0, 1])


def test_polygon_loop_orientation_and_axis_field_converge_to_circular_reference():
    from emlab.core import MU0

    radius, height, current = 0.1, 0.05, 2.0
    loop = m.circular_loop(radius, 256)
    field = m.polyline_field([0, 0, height], loop, current)
    exact = MU0 * current * radius**2 / (2 * (radius**2 + height**2) ** 1.5)
    assert field[2] == pytest.approx(exact, rel=1e-4)
    assert np.max(abs(field[:2])) < 1e-18
    assert np.allclose(m.polyline_field([0, 0, height], loop[::-1], current), -field)


def test_loop_center_and_far_axis_have_correct_limiting_scaling():
    from emlab.core import MU0

    radius = 0.1
    assert m.loop_axis(radius, 0, 3) == pytest.approx(MU0 * 3 / (2 * radius))
    assert m.loop_axis(radius, 10) / m.loop_axis(radius, 20) == pytest.approx(8, rel=2e-4)


def test_off_axis_elliptic_field_matches_independent_polygon_filament_integration():
    points = np.array([[0.03, 0.02, 0.04], [0.14, 0, 0.02], [0, 0, 0.05]])
    exact = m.circular_field(points, 0.1, 2)
    polygon = m.polyline_field(points, m.circular_loop(0.1, 512), 2)
    from emlab.core import relative_error

    assert relative_error(exact, polygon) < 5e-5


def test_vector_potential_curl_matches_biot_savart_field():
    point = np.array([0.7, 0.4, 0.2])
    start, end = [0, 0, -1], [0, 0, 1]
    step = 1e-5
    derivative = np.empty((3, 3))
    for axis in range(3):
        offset = np.eye(3)[axis] * step
        derivative[:, axis] = (
            m.segment_potential(point + offset, start, end) - m.segment_potential(point - offset, start, end)
        ) / (2 * step)
    curl = np.array(
        [
            derivative[2, 1] - derivative[1, 2],
            derivative[0, 2] - derivative[2, 0],
            derivative[1, 0] - derivative[0, 1],
        ]
    )
    assert np.allclose(curl, m.segment_field(point, start, end), rtol=1e-8, atol=1e-18)


def test_neumann_mutual_inductance_obeys_reciprocity_and_winding_orientation():
    first = m.circular_loop(0.1, 48)
    second = m.circular_loop(0.08, 48, (0, 0, 0.05))
    mutual = m.mutual_inductance(first, second)
    assert mutual > 0
    assert m.mutual_inductance(second, first) == pytest.approx(mutual, rel=1e-12)
    assert m.mutual_inductance(first[::-1], second) == pytest.approx(-mutual, rel=1e-12)


def test_polygon_neumann_integral_converges_to_coaxial_elliptic_solution():
    exact = m.coaxial_mutual(0.1, 0.08, 0.05)
    errors = []
    for segments in (24, 48, 96):
        first = m.circular_loop(0.1, segments)
        second = m.circular_loop(0.08, segments, (0, 0, 0.05))
        errors.append(abs(m.mutual_inductance(first, second) - exact))
    assert errors[2] < errors[1] < errors[0]
    assert errors[-1] / exact < 0.001


def test_internal_flux_term_is_separate_from_external_thin_wire_inductance():
    from emlab.core import MU0

    external = m.loop_inductance(0.1, 0.001)
    uniform_current = m.loop_inductance(0.1, 0.001, internal_flux=True)
    assert external > 0
    assert uniform_current - external == pytest.approx(MU0 * 0.1 / 4)
    with pytest.raises(ValueError, match="thin wire"):
        m.loop_inductance(0.1, 0.02)


def test_coupled_coil_energy_detects_impossible_mutual_coupling():
    matrix = 1e-6 * np.array([[2, 0.8], [0.8, 1]])
    flux, energy = m.coupled_energy(matrix, [2, -1])
    assert np.allclose(flux, matrix @ [2, -1])
    assert energy == pytest.approx(0.5 * np.array([2, -1]) @ matrix @ [2, -1])
    with pytest.raises(ValueError):
        m.coupled_energy(1e-6 * np.array([[1, 2], [2, 1]]), [1, 1])


def test_magnetostatic_vector_potential_matches_manufactured_current_and_energy():
    from emlab.core import MU0, relative_error
    from emlab.grid import dual_weights, from_bounds

    grid = from_bounds((0, 1), (0, 1), 41, 41)
    x, y = grid.mesh()
    exact = 1e-5 * np.sin(np.pi * x) * np.sin(np.pi * y)
    current = 2 * np.pi**2 * exact / (MU0 * 3)
    result = m.solve_az(grid, current, 3)
    assert relative_error(exact, result["az_wb_m"]) < 0.001
    work = 0.5 * np.sum(current * result["az_wb_m"] * dual_weights(grid))
    assert result["energy_j_m"] == pytest.approx(work, rel=1e-12)


def test_coaxial_coils_attract_with_parallel_currents_and_repel_when_reversed():
    force = m.coaxial_force(0.1, 0.08, 0.05, 3, 2)
    assert force < 0
    assert m.coaxial_force(0.1, 0.08, 0.05, 3, -2) == pytest.approx(-force)
    finer = m.coaxial_force(0.1, 0.08, 0.05, 3, 2, step_m=1e-6)
    assert force == pytest.approx(finer, rel=1e-6)
