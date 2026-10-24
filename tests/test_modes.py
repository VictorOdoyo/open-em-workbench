import numpy as np
import pytest

from emlab import modes as m


def test_rectangular_cutoff_matches_standard_te10_and_excludes_spurious_tem():
    from emlab.core import C0

    assert m.rectangular_cutoff(0.02, 0.01, 1, 0) == pytest.approx(C0 / 0.04)
    with pytest.raises(ValueError):
        m.rectangular_cutoff(0.02, 0.01, 0, 0)
    with pytest.raises(ValueError):
        m.rectangular_cutoff(0.02, 0.01, 1, 0, "tm")


def test_generalized_cavity_modes_match_independent_rectangular_cutoffs():
    from emlab.grid import from_bounds

    grid = from_bounds((0, 0.02), (0, 0.01), 33, 25)
    for polarization, indices in (("tm", (1, 1)), ("te", (1, 0))):
        result = m.cavity_modes(grid, epsilon_r=4, polarization=polarization, count=3)
        expected = m.rectangular_cutoff(0.02, 0.01, *indices, polarization, epsilon_r=4)
        assert result["cutoff_hz"][0] == pytest.approx(expected, rel=0.002)
        assert np.max(result["residuals"]) < 1e-8
        assert np.all(np.diff(result["cutoff_hz"]) >= 0)
