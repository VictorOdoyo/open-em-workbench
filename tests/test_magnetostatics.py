import numpy as np
import pytest

from emlab import magnetostatics as m


def test_finite_wire_field_matches_closed_form_and_right_hand_rule():
    from emlab.core import MU0

    field = m.segment_field([[1, 0, 0], [0, 0, 3]], [0, 0, -1], [0, 0, 1], 2)
    assert field[0, 1] == pytest.approx(MU0 * 2 / (4 * np.pi) * np.sqrt(2))
    assert np.allclose(field[0, [0, 2]], 0)
    assert np.allclose(field[1], 0)
    with pytest.raises(ValueError, match="singular"):
        m.segment_field([0, 0, 0], [0, 0, -1], [0, 0, 1])


def test_polygon_loop_orientation_and_axis_field_converge_to_circular_reference():
    from emlab.core import MU0

    radius, height, current = 0.1, 0.05, 2.0
    loop = m.circular_loop(radius, 256)
    field = m.polyline_field([0, 0, height], loop, current)
    exact = MU0 * current * radius**2 / (2 * (radius**2 + height**2) ** 1.5)
    assert field[2] == pytest.approx(exact, rel=1e-4)
    assert np.max(abs(field[:2])) < 1e-18
    assert np.allclose(m.polyline_field([0, 0, height], loop[::-1], current), -field)


def test_loop_center_and_far_axis_have_correct_limiting_scaling():
    from emlab.core import MU0

    radius = 0.1
    assert m.loop_axis(radius, 0, 3) == pytest.approx(MU0 * 3 / (2 * radius))
    assert m.loop_axis(radius, 10) / m.loop_axis(radius, 20) == pytest.approx(8, rel=2e-4)
