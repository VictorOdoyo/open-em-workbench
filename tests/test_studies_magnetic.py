import numpy as np
import pytest

from emlab import studies_magnetic as m


def test_coil_study_converges_to_elliptic_reference_and_has_attractive_force():
    result = m.coil_mutual()
    assert result["summary"]["finest_relative_error"] < 0.001
    assert result["summary"]["coaxial_attractive_force_n"] < 0
    assert np.any(result["arrays"]["excluded"])


def test_current_field_study_matches_ampere_and_magnetic_work():
    summary = m.current_fields()["summary"]
    assert abs(summary["net_current_a"]) < 1e-12
    assert summary["ampere_relative_error"] < 0.03
    assert summary["field_energy_j_m"] == pytest.approx(summary["source_work_j_m"], rel=1e-11)
