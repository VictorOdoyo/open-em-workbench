import pytest

from emlab import layers as m


def test_interface_enforces_tangential_electric_and_magnetic_continuity():
    scattering = m.interface(1 / 50, 1 / 100)
    r, t = scattering[0, 0], scattering[1, 0]
    assert 1 + r == pytest.approx(t)
    assert (1 - r) / 50 == pytest.approx(t / 100)
