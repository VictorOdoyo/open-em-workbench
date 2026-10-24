from emlab import studies_design as m


def test_sensitivity_study_resolves_each_perturbed_field():
    result = m.sensitivity()
    assert result["summary"]["maximum_relative_error"] < 1e-5
    assert len(result["tables"]["samples"]) == 6


def test_force_study_preserves_sign_and_converges():
    result = m.force()
    assert result["summary"]["finest_force_y_n_m"] < 0
    rows = result["tables"]["meshes"]
    assert rows[-1]["relative_error"] < rows[0]["relative_error"] / 10
    assert rows[-1]["relative_error"] < 2e-4
