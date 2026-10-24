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
