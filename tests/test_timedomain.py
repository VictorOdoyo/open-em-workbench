import numpy as np
import pytest

from emlab import timedomain as m


def test_courant_limit_includes_dimension_and_fastest_material():
    from emlab.core import C0

    assert m.courant_step([0.001], safety=1) == pytest.approx(0.001 / C0)
    assert m.courant_step([0.001, 0.001], safety=1) == pytest.approx(0.001 / (C0 * np.sqrt(2)))
    assert m.courant_step([0.001], epsilon_min=4, safety=1) == pytest.approx(0.002 / C0)


def test_conductive_update_matches_trapezoidal_ohmic_decay():
    from emlab.core import EPS0

    dt, conductivity = 1e-12, 0.1
    a, b = m.electric_coefficients(3, conductivity, dt)
    assert abs(a) <= 1
    assert a == pytest.approx((3 * EPS0 - conductivity * dt / 2) / (3 * EPS0 + conductivity * dt / 2))
    assert b == pytest.approx(dt / (3 * EPS0 + conductivity * dt / 2))
    assert m.electric_coefficients(1, 0, dt)[0] == 1


def test_pml_profile_is_symmetric_zero_interior_and_correctly_staggered():
    coordinate = np.linspace(0, 1, 101)
    sigma, kappa, alpha = m.pml_profile(coordinate, 0, 1, 0.1, alpha_max_s=1e8)
    assert np.allclose(sigma, sigma[::-1])
    assert np.all(sigma[11:90] == 0)
    assert kappa[0] == 3 and kappa[50] == 1
    assert alpha[0] == 0 and alpha[5] > 0
    shifted = m.pml_profile((coordinate[:-1] + coordinate[1:]) / 2, 0, 1, 0.1)[0]
    assert shifted[0] < sigma[0]


def test_cpml_discrete_memory_approaches_continuous_coordinate_stretch():
    sigma, kappa, alpha, omega, dt = 2e8, 2.0, 3e7, 1e9, 1e-14
    decay, memory, _ = m.pml_coefficients(sigma, kappa, alpha, dt)
    discrete = 1 / kappa + memory / (1 - decay * np.exp(-1j * omega * dt))
    continuous = 1 / (kappa + sigma / (alpha + 1j * omega))
    assert discrete == pytest.approx(continuous, rel=1e-5)
    assert m.pml_coefficients(0, 1, 0, dt)[:2] == (1, 0)


def test_one_dimensional_unit_courant_has_exact_linear_dispersion():
    from emlab.core import C0

    spacing = 0.01
    k = np.linspace(1, 0.9 * np.pi / spacing, 21)[:, None]
    omega = m.dispersion_frequency(k, [spacing], spacing / C0, C0)
    assert np.allclose(omega, C0 * k[:, 0])


def test_discrete_wavenumber_inverts_frequency_and_rejects_grid_stopband():
    from emlab.core import C0

    spacing, dt = 0.01, 0.005 / C0
    f = np.array([1e8, 1e9])
    k = m.numerical_wavenumber(f, spacing, dt, C0)
    recovered = m.dispersion_frequency(k[:, None], [spacing], dt, C0) / (2 * np.pi)
    assert np.allclose(f, recovered)
    with pytest.raises(ValueError):
        m.numerical_wavenumber(0.49 / dt, spacing, dt, C0)
