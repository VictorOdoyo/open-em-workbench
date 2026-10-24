import numpy as np
import pytest

from emlab import geometry as m


def test_shape_masks_follow_physical_coordinates():
    from emlab.grid import from_bounds

    grid = from_bounds((-1, 1), (-1, 1), 21, 21)
    disk = m.circle(grid, (0, 0), 0.5)
    box = m.rectangle(grid, (-0.2, 0.2), (-0.2, 0.2))
    assert disk[10, 10] and not disk[0, 0]
    assert np.all(~box | disk)


def test_annular_shell_excludes_center_and_has_correct_signed_distance():
    from emlab.grid import from_bounds

    grid = from_bounds((-1, 1), (-1, 1), 21, 21)
    shell = m.annulus(grid, (0, 0), 0.2, 0.6)
    assert not shell[10, 10] and shell[10, 14]
    assert m.circle_distance(grid, (0, 0), 0.6)[10, 10] == pytest.approx(-0.6)
