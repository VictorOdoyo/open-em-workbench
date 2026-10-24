import numpy as np
import pytest

from emlab import scattering as m


def test_pec_cylindrical_series_cancels_incident_electric_field_on_the_boundary():
    radius = 0.04
    angle = np.linspace(0, 2 * np.pi, 121)
    points = radius * np.column_stack((np.cos(angle), np.sin(angle)))
    coefficients = m.pec_coefficients(1e9, radius)
    for incidence in (0, 0.7):
        fields = m.exterior_field(coefficients, points, incidence)
        assert np.max(abs(fields["total"])) < 1e-12


def test_scattering_width_integrates_to_modal_sum_and_satisfies_optical_theorem():
    angles = np.linspace(0, 2 * np.pi, 2049)
    result = m.widths(m.pec_coefficients(2e9, 0.04), angles)
    assert np.trapezoid(result["differential_m_rad"], angles) == pytest.approx(result["scattering_m"])
    assert result["scattering_m"] == pytest.approx(result["extinction_m"], rel=1e-12)
    assert abs(result["absorption_m"]) < 1e-13


def test_dielectric_cylinder_no_contrast_unitarity_and_absorptive_passivity():
    empty = m.dielectric_coefficients(1e9, 0.04, 1)
    assert np.max(abs(empty["coefficient"])) < 1e-14
    lossless = m.widths(m.dielectric_coefficients(1e9, 0.04, 4), [0, 1])
    assert lossless["extinction_m"] == pytest.approx(lossless["scattering_m"], rel=1e-12)
    lossy = m.widths(m.dielectric_coefficients(1e9, 0.04, 4 - 0.5j), [0, 1])
    assert lossy["absorption_m"] > 0


def test_frequency_domain_pec_scattering_agrees_with_cylinder_reference_with_staircase_tolerance():
    from emlab.fdfd import scatter
    from emlab.geometry import circle
    from emlab.grid import from_bounds, sample_bilinear

    grid = from_bounds((-0.2, 0.2), (-0.2, 0.2), 101, 101)
    radius = 0.04
    result = scatter(grid, 0.8e9, pec_mask=circle(grid, (0, 0), radius), pml_cells=14)
    angles = np.linspace(0, 2 * np.pi, 17)[:-1]
    points = 0.09 * np.column_stack((np.cos(angles), np.sin(angles)))
    numerical = np.array([sample_bilinear(grid, result["scattered"], x, y) for x, y in points])
    exact = m.exterior_field(m.pec_coefficients(0.8e9, radius), points)["scattered"]
    comparison = m.compare_samples(exact, numerical)
    assert comparison["relative_l2"] < 0.2
