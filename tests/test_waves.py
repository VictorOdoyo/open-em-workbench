import pytest

from emlab import waves as m


def test_fresnel_air_to_dielectric_obeys_power_conservation_not_amplitude_sum():
    from emlab.core import ETA0

    result = m.fresnel_normal(ETA0, ETA0 / 2)
    assert result["r"] == pytest.approx(-1 / 3)
    assert result["t"] == pytest.approx(2 / 3)
    assert result["R"] + result["T"] == pytest.approx(1)
    assert abs(result["r"]) + abs(result["t"]) == pytest.approx(1)
