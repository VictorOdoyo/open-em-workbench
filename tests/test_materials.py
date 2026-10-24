import numpy as np
import pytest

from emlab import materials as m


def test_passive_conductivity_has_negative_imaginary_permittivity():
    from emlab.core import EPS0

    eps = m.permittivity(1e9, 4, 0.1)
    assert eps.real == 4 and eps.imag < 0
    assert -eps.imag * (2 * np.pi * 1e9) * EPS0 == pytest.approx(0.1)
    with pytest.raises(ValueError):
        m.permittivity(0)


def test_propagation_decays_and_satisfies_maxwell_impedance_relation():
    from emlab.core import C0, ETA0

    gamma, eta = m.propagation(1e9, 4)
    assert gamma == pytest.approx(1j * 4 * np.pi * 1e9 / C0)
    assert eta == pytest.approx(ETA0 / 2)
    lossy, impedance = m.propagation(1e9, m.permittivity(1e9, 4, 1))
    assert lossy.real > 0 and impedance.real > 0
    evanescent, _ = m.propagation(1e9, -4)
    assert evanescent.real > 0 and abs(evanescent.imag) < 1e-12


def test_debye_limits_and_relaxation_loss_peak():
    tau = 1e-9
    values = m.debye([0, 1 / (2 * np.pi * tau), 1e20], 8, 2, tau)
    assert values[0] == pytest.approx(8)
    assert values[1] == pytest.approx(5 - 3j)
    assert values[2].real == pytest.approx(2)


def test_lorentz_static_strength_and_resonance_sign():
    assert m.lorentz(0, 2, 3, 1e9, 1e8) == pytest.approx(5)
    assert m.lorentz(1e9, 2, 3, 1e9, 1e8) == pytest.approx(2 - 30j)


def test_drude_is_metallic_below_plasma_and_passive():
    eps = m.drude(np.array([0.1, 10]) * 1e12, 1, 1e12, 1e10)
    assert eps[0].real < 0 and eps[1].real > 0
    assert np.all(eps.imag < 0)
    gamma, _ = m.propagation(0.1e12, eps[0])
    assert gamma.real > 0


def test_loss_tangent_equivalent_conductivity_is_frequency_specific():
    sigma = m.loss_conductivity(2e9, 4, 0.02)
    eps = m.permittivity(2e9, 4, sigma)
    assert -eps.imag / eps.real == pytest.approx(0.02)


def test_copper_skin_depth_and_impedance_match_good_conductor_limit():
    depth = m.skin_depth(1e6, 5.8e7)
    assert 65e-6 < depth < 67e-6
    _, exact = m.propagation(1e6, m.permittivity(1e6, 1, 5.8e7))
    assert m.surface_impedance(1e6, 5.8e7) == pytest.approx(exact, rel=1e-8)


def test_rotated_anisotropy_preserves_principal_energy():
    angle = 0.4
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    tensor = m.rotated_tensor([2, 6], rotation)
    assert np.allclose(np.linalg.eigvalsh(tensor), [2, 6])
    assert tensor[0, 1] != 0
