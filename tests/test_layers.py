import numpy as np
import pytest

from emlab import layers as m


def test_interface_enforces_tangential_electric_and_magnetic_continuity():
    scattering = m.interface(1 / 50, 1 / 100)
    r, t = scattering[0, 0], scattering[1, 0]
    assert 1 + r == pytest.approx(t)
    assert (1 - r) / 50 == pytest.approx(t / 100)


def test_layer_propagation_handles_optically_thick_absorbers_without_overflow():
    result = m.propagation(3 - 1e6j, 1)
    assert np.all(np.isfinite(result)) and np.all(result == 0)
    vacuum = m.propagation(2, 0.3)
    assert vacuum[1, 0] == pytest.approx(np.exp(-0.6j))


def test_empty_oblique_stack_reduces_to_independent_fresnel_formula():
    from emlab.waves import fresnel_oblique

    for polarization in ("te", "tm"):
        system = m.components(1e9, [], [], epsilon_substrate=4, angle_radians=0.4, polarization=polarization)
        interface = system["pieces"][0]
        expected = fresnel_oblique(1e9, 1, 4, 0.4, polarization)
        assert interface[0, 0] == pytest.approx(expected["r"])
        assert interface[1, 0] == pytest.approx(expected["t"])
