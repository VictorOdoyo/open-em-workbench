import numpy as np

from emlab import fdfd as m


def test_frequency_domain_pml_has_negative_imaginary_stretch_and_real_interior():
    from emlab.grid import Grid2D

    grid = Grid2D(51, 41, 0.005, 0.005)
    stretch = m.coordinate_stretch(grid, 1e9, 8)
    assert stretch["xe"].shape == (51,) and stretch["xh"].shape == (50,)
    assert stretch["xe"][0].imag < 0
    assert stretch["xe"][25] == 1
    assert np.allclose(stretch["ye"], stretch["ye"][::-1])
