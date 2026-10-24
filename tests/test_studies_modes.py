import numpy as np

from emlab import studies_modes as m


def test_multi_family_cutoff_study_has_no_spurious_tem_mode():
    result = m.cavity_cutoffs()
    assert result["summary"]["maximum_relative_cutoff_error"] < 0.003
    assert result["summary"]["maximum_eigen_residual"] < 1e-8
    assert all(row["reference_hz"] > 0 for row in result["tables"]["cutoffs"])


def test_slab_guidance_study_retains_a_resolved_bound_mode_and_reports_actual_error():
    result = m.slab_guidance()
    assert result["summary"]["finest_relative_error"] < 3e-5
    assert all(row["guided_modes_found"] == 1 for row in result["tables"]["meshes"])


def test_perturbation_study_reports_downward_dielectric_frequency_shifts():
    result = m.mode_perturbation()
    assert np.all(result["arrays"]["measured_percent"][1:] < 0)
    assert result["summary"]["first_nonzero_prediction_relative_error"] < 0.05
    assert result["summary"]["largest_absolute_fractional_shift_error"] > 0
