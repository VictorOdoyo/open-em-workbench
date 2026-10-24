import numpy as np
import pytest

from emlab import fdtd1d as m


def test_yee_state_has_explicit_space_time_staggering_and_courant_guard():
    from emlab.core import C0

    state = m.Yee1D(101, 0.001)
    assert state.e.shape == (101,) and state.h.shape == (100,)
    assert state.magnetic_time_s == -state.dt / 2
    with pytest.raises(ValueError, match="Courant"):
        m.Yee1D(101, 0.001, dt_s=0.002 / C0)
    fast = m.Yee1D(101, 0.001, epsilon_r=0.25)
    assert fast.dt == pytest.approx(state.dt / 2)


def test_yee_impulse_update_matches_independent_three_point_wave_stencil():
    state = m.Yee1D(7, 0.001, courant=0.5)
    state.e[3] = 1
    state.step()
    assert np.allclose(state.e, [0, 0, 0.25, 0.5, 0.25, 0, 0])
    assert state.iteration == 1


def test_impressed_current_has_ampere_sign_and_si_scaling():
    from emlab.core import EPS0

    state = m.Yee1D(7, 0.001)
    current = np.zeros(7)
    current[3] = 0.1
    state.step(current)
    assert state.e[3] == pytest.approx(-state.dt / EPS0 * 0.1)
    assert state.electric_time_s - state.magnetic_time_s == pytest.approx(state.dt / 2)
