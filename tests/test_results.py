import numpy as np
import pytest

from emlab import results as m


def test_study_schema_retains_complex_fields_but_rejects_nonfinite_and_unaligned_plots():
    result = m.make(
        "test-study",
        {"frequency_hz": 1e9},
        {"error": 0.1},
        {"x": [1.0, 2], "field": [1j, 2j], "magnitude": [1.0, 2]},
        [m.curve("Field magnitude", "x", [("magnitude", "numeric")], "x (m)", "|E| (V/m)")],
    )
    assert np.iscomplexobj(result["arrays"]["field"])
    result["arrays"]["magnitude"] = np.array([1.0])
    with pytest.raises(ValueError, match="aligned"):
        m.validate(result)
    result["arrays"]["magnitude"] = np.array([np.nan, 2])
    with pytest.raises(ValueError, match="finite"):
        m.validate(result)


def test_complex_curve_abscissas_are_not_silently_cast_to_real():
    with pytest.raises(ValueError, match="coordinates"):
        m.make(
            "invalid-curve",
            {},
            {"value": 1},
            {"x": [1 + 1j, 2 + 2j], "y": [1, 2]},
            [m.curve("invalid", "x", [("y", "values")], "x", "y")],
        )
