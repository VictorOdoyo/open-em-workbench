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
