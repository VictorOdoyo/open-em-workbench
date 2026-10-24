import pytest

from emlab import modes as m


def test_rectangular_cutoff_matches_standard_te10_and_excludes_spurious_tem():
    from emlab.core import C0

    assert m.rectangular_cutoff(0.02, 0.01, 1, 0) == pytest.approx(C0 / 0.04)
    with pytest.raises(ValueError):
        m.rectangular_cutoff(0.02, 0.01, 0, 0)
    with pytest.raises(ValueError):
        m.rectangular_cutoff(0.02, 0.01, 1, 0, "tm")
