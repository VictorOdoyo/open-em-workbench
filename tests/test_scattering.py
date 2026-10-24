import numpy as np

from emlab import scattering as m


def test_pec_cylindrical_series_cancels_incident_electric_field_on_the_boundary():
    radius = 0.04
    angle = np.linspace(0, 2 * np.pi, 121)
    points = radius * np.column_stack((np.cos(angle), np.sin(angle)))
    coefficients = m.pec_coefficients(1e9, radius)
    for incidence in (0, 0.7):
        fields = m.exterior_field(coefficients, points, incidence)
        assert np.max(abs(fields["total"])) < 1e-12
