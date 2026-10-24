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
