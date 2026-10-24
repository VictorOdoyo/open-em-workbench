import numpy as np
import pytest

from emlab import waves as m


def test_fresnel_air_to_dielectric_obeys_power_conservation_not_amplitude_sum():
    from emlab.core import ETA0

    result = m.fresnel_normal(ETA0, ETA0 / 2)
    assert result["r"] == pytest.approx(-1 / 3)
    assert result["t"] == pytest.approx(2 / 3)
    assert result["R"] + result["T"] == pytest.approx(1)
    assert abs(result["r"]) + abs(result["t"]) == pytest.approx(1)


def test_longitudinal_branch_decays_above_total_internal_reflection_threshold():
    from emlab.core import C0

    k0 = 2 * np.pi * 1e9 / C0
    kz = m.longitudinal_wavenumber(1e9, 1, transverse_wavenumber=1.5 * k0)
    assert kz.imag < 0
    assert abs(np.exp(-1j * kz * 1)) < 1
    assert m.longitudinal_wavenumber(1e9, 4, transverse_wavenumber=k0) == pytest.approx(np.sqrt(3) * k0)


def test_oblique_brewster_zero_and_total_internal_reflection():
    angle = np.arctan(2)
    tm = m.fresnel_oblique(1e9, 1, 4, angle, "tm")
    assert abs(tm["r"]) < 1e-14
    for pol in ("te", "tm"):
        result = m.fresnel_oblique(1e9, 4, 1, np.pi / 3, pol)
        assert result["R"] == pytest.approx(1)
        assert result["T"] == pytest.approx(0)
