import numpy as np
import pytest

from emlab import capacitance as m


def test_analytic_capacitances_scale_with_dielectric_and_geometry():
    from emlab.core import EPS0

    assert m.parallel_plate(2, 1, 3) == pytest.approx(6 * EPS0)
    assert m.coaxial(1, np.e, 4) == pytest.approx(8 * np.pi * EPS0)
    assert m.coaxial(2, 6) == pytest.approx(m.coaxial(1, 3))


def test_extracted_capacitance_has_reciprocity_gauge_nullspace_and_positive_energy():
    from emlab.geometry import conductor_labels
    from emlab.grid import Grid2D

    grid = Grid2D(25, 21, 0.001, 0.001)
    first = np.zeros(grid.shape, bool)
    second = first.copy()
    first[8:13, 7:9] = True
    second[8:13, 16:18] = True
    result = m.extract(grid, conductor_labels(grid, [first, second]), 3)
    matrix = result["matrix"]
    assert np.allclose(matrix, matrix.T, atol=1e-24, rtol=1e-12)
    assert np.max(abs(matrix.sum(axis=0))) < 1e-24
    assert np.all(np.linalg.eigvalsh(matrix[1:, 1:]) > 0)
    assert matrix[1, 2] < 0
    assert np.allclose(np.sum(result["basis"], axis=0), 1)


def test_terminal_energy_is_gauge_invariant_and_rejects_active_capacitance():
    matrix = 2e-11 * np.array([[1, -1], [-1, 1]])
    charge, energy = m.terminal_response(matrix, [0, 3])
    shifted, shifted_energy = m.terminal_response(matrix, [5, 8])
    assert np.allclose(charge, shifted, atol=1e-25)
    assert energy == pytest.approx(shifted_energy)
    assert energy == pytest.approx(0.5 * 2e-11 * 9)
    with pytest.raises(ValueError, match="passive"):
        m.terminal_response(-matrix, [0, 3])


def test_floating_plate_reproduces_series_capacitor_voltage_division():
    matrix = 1e-12 * np.array([[2, -2, 0], [-2, 5, -3], [0, -3, 3]])
    voltages, charges = m.floating_voltages(matrix, [0, 2], [0, 10])
    assert voltages[1] == pytest.approx(6)
    assert charges[1] == pytest.approx(0, abs=1e-26)
    assert charges[2] == pytest.approx(12e-12)


def test_common_differential_transform_preserves_quadratic_field_energy():
    matrix = 1e-11 * np.array([[3, -1], [-1, 3]])
    transform = np.array([[1.0, 0.5], [1, -0.5]])
    modal = m.transform(matrix, transform)
    voltage = np.array([2.0, 3.0])
    assert m.terminal_response(matrix, transform @ voltage)[1] == pytest.approx(
        m.terminal_response(modal, voltage)[1]
    )
    assert abs(modal[0, 1]) < 1e-25


def test_fixed_voltage_force_attracts_parallel_plates():
    gap, step, voltage = 0.002, 1e-6, 5
    minus = np.array([[m.parallel_plate(0.01, gap - step)]])
    plus = np.array([[m.parallel_plate(0.01, gap + step)]])
    force = m.coenergy_force(minus, plus, [voltage], step)
    from emlab.core import EPS0

    assert force < 0
    assert force == pytest.approx(-0.5 * EPS0 * 0.01 * voltage**2 / gap**2, rel=1e-6)


def test_terminal_response_rejects_small_but_physical_nonreciprocity():
    matrix = 1e-12 * np.array([[2.0, -1.0], [-1.000001, 2.0]])
    with pytest.raises(ValueError, match="reciprocal"):
        m.terminal_response(matrix, [1.0, 0.0])
