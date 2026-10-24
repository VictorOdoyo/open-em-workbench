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


def test_lossless_film_conserves_power_and_resonant_halfwave_is_transparent():
    from emlab.core import C0

    frequency = np.linspace(0.5e9, 1.5e9, 31)
    result = m.stack(frequency, [4], [C0 / (4 * 1e9)])
    assert np.allclose(result["R"] + result["T"], 1)
    center = m.stack(1e9, [4], [C0 / (4 * 1e9)])
    assert center["R"] < 1e-25
    assert center["T"] == pytest.approx(1)
    thick = m.stack(1e9, [-1000 - 1e5j], [10])
    assert np.isfinite(thick["A"]) and 0 <= thick["A"] <= 1


def test_internal_layer_fields_match_external_boundary_conditions():
    from emlab.core import C0, ETA0

    thickness = C0 / (7 * 1e9)
    result = m.stack(1e9, [3 - 0.2j], [thickness])
    field = m.fields(1e9, [3 - 0.2j], [thickness], 0, [0, thickness])
    assert field["electric"][0] == pytest.approx(1 + result["r"])
    assert field["magnetic"][0] == pytest.approx((1 - result["r"]) / ETA0)
    assert field["electric"][-1] == pytest.approx(result["t"])
    assert field["magnetic"][-1] == pytest.approx(result["t"] / ETA0)


def test_layer_resolved_absorption_sums_to_global_power_loss():
    epsilons = [2 - 0.4j, 5 - 0.7j, 3]
    thickness = [0.02, 0.03, 0.01]
    result = m.stack(2e9, epsilons, thickness)
    absorbed = m.absorption_by_layer(2e9, epsilons, thickness)
    assert np.all(absorbed > -1e-13)
    assert abs(absorbed[-1]) < 1e-13
    assert absorbed.sum() == pytest.approx(result["A"], abs=1e-12)


def test_quarter_wave_coating_cancels_design_frequency_reflection():
    coating = m.antireflection(3e9, 1, 2)
    result = m.stack(3e9, [coating["epsilon_r"]], [coating["thickness_m"]], epsilon_substrate=4)
    assert result["R"] < 1e-25
    assert result["T"] == pytest.approx(1)


def test_bragg_stopband_reflection_increases_with_pair_count():
    results = []
    for pairs in (1, 4):
        epsilon, thickness = m.bragg_mirror(10e9, 3, 1.5, pairs)
        results.append(m.stack(10e9, epsilon, thickness)["R"])
    assert results[1] > results[0]
    assert results[1] > 0.98
