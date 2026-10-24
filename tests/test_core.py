import numpy as np
import pytest

from emlab import core as m


def test_vacuum_constants_and_frequency_wavelength_identity():
    assert m.EPS0 * m.MU0 * m.C0**2 == pytest.approx(1)
    assert m.ETA0 == pytest.approx(376.7303134)
    assert m.wavelength(1e9) == pytest.approx(0.299792458)
    assert np.allclose(m.frequency(m.wavelength([1e6, 1e9])), [1e6, 1e9])
    with pytest.raises(ValueError):
        m.wavelength(0)


def test_finite_validation_retains_complex_phase_and_rejects_invalid_inputs():
    assert m.finite_array([1j, 2]).dtype.kind == "c"
    for value in ([np.inf], ["1"], [None]):
        with pytest.raises(ValueError):
            m.finite_array(value)
    with pytest.raises(ValueError):
        m.finite_array([1j], real=True)


def test_field_and_power_decibels_do_not_confuse_amplitude_with_energy():
    assert m.to_db(0.5, field=True) == pytest.approx(m.to_db(0.25))
    assert m.to_db(1j, field=True) == 0
    assert np.isneginf(m.to_db(0))
    assert np.allclose(m.from_db(m.to_db([0.1, 1, 10])), [0.1, 1, 10])
    with pytest.raises(ValueError):
        m.to_db(-1)


def test_relative_error_preserves_amplitude_and_complex_phase_mistakes():
    assert m.relative_error([1, 2], [2, 4]) == pytest.approx(1)
    assert m.relative_error([1j, 2j], [-1j, -2j]) == pytest.approx(2)
    with pytest.raises(ValueError):
        m.relative_error([0], [1])


def test_named_random_streams_are_stable_and_distinct():
    assert np.array_equal(
        m.random_stream(9, "source").normal(size=8), m.random_stream(9, "source").normal(size=8)
    )
    assert not np.array_equal(
        m.random_stream(9, "source").normal(size=8), m.random_stream(9, "noise").normal(size=8)
    )
