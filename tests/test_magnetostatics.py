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
