from emlab import studies_design as m


def test_sensitivity_study_resolves_each_perturbed_field():
    result = m.sensitivity()
    assert result["summary"]["maximum_relative_error"] < 1e-5
    assert len(result["tables"]["samples"]) == 6
