import numpy as np

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


def test_design_study_improves_energy_with_fixed_material_budget():
    result = m.design()
    assert result["summary"]["capacitance_gain_fraction"] > 0.05
    assert result["summary"]["material_mean_defect"] < 1e-12
    assert np.all(np.diff(result["arrays"]["energy_pj_m"]) >= 0)


def test_uncertainty_study_distinguishes_mean_precision_and_unit_spread():
    result = m.uncertainty()
    summary = result["summary"]
    assert summary["std_relative_difference"] < 0.15
    assert (
        summary["mean_ci95_high_f_m"] - summary["mean_ci95_low_f_m"]
        < (summary["population_p975_f_m"] - summary["population_p025_f_m"]) / 10
    )
    assert np.array_equal(result["arrays"]["samples"], m.uncertainty()["arrays"]["samples"])
