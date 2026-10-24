import numpy as np
import pytest

from emlab import fdtd2d as m


def test_tmz_update_matches_five_point_wave_stencil_and_field_orientations():
    from emlab.grid import Grid2D

    state = m.TMz2D(Grid2D(9, 9, 0.001, 0.001), courant=0.5)
    state.ez[4, 4] = 1
    state.step()
    assert state.ez[4, 4] == pytest.approx(0.5)
    assert state.ez[4, 3] == pytest.approx(0.125)
    assert state.ez[3, 4] == pytest.approx(0.125)
    assert state.hx[3, 4] < 0 and state.hy[4, 3] > 0


def test_embedded_pec_enforces_zero_electric_field_and_rejects_internal_sources():
    from emlab.grid import Grid2D

    grid = Grid2D(9, 9, 0.001, 0.001)
    mask = np.zeros(grid.shape, bool)
    mask[4, 4] = True
    state = m.TMz2D(grid, pec_mask=mask)
    state.ez[4, 4] = 1
    state.step()
    assert np.max(abs(state.ez)) == 0
    with pytest.raises(ValueError, match="PEC"):
        state.step(mask.astype(float))
