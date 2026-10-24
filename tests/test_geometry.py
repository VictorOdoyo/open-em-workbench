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


def test_conductor_assignment_detects_contacts_and_overlaps():
    from emlab.grid import Grid2D

    grid = Grid2D(9, 9, 1, 1)
    mask = np.zeros(grid.shape, bool)
    mask[3:5, 3:5] = True
    labels = m.conductor_labels(grid, [mask])
    assert labels[3, 3] == 1 and labels[0, 0] == 0 and labels[1, 1] == -1
    with pytest.raises(ValueError, match="overlap"):
        m.conductor_labels(grid, [mask, mask])


def test_material_painting_has_explicit_ordered_precedence():
    from emlab.grid import Grid2D

    grid = Grid2D(9, 9, 1, 1)
    whole = np.ones(grid.shape, bool)
    center = np.zeros(grid.shape, bool)
    center[4, 4] = True
    values = m.material_map(grid, 1, [(whole, 2), (center, 5 - 0.1j)])
    assert values[4, 4] == 5 - 0.1j and values[3, 3] == 2


def test_subpixel_fill_integrates_circle_area_and_clips_boundary_cells():
    from emlab.grid import dual_weights, from_bounds

    grid = from_bounds((-1, 1), (-1, 1), 25, 25)
    fill = m.fill_fraction(grid, lambda x, y: x * x + y * y < 0.6**2, 16)
    area = np.sum(fill * dual_weights(grid))
    assert area == pytest.approx(np.pi * 0.6**2, rel=0.003)
    assert np.any((fill > 0) & (fill < 1))
