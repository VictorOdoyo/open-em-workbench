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
