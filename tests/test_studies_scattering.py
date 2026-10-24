from emlab import studies_scattering as m


def test_line_source_study_matches_complex_green_function_in_the_open_interior():
    summary = m.line_source()["summary"]
    assert summary["complex_field_relative_l2"] < 0.015
    assert summary["solver_residual"] < 1e-10


def test_cylinder_study_reports_real_mesh_error_and_lossless_reference_width():
    result = m.cylinder()
    assert result["summary"]["finest_complex_relative_l2"] < 0.15
    assert result["summary"]["analytic_optical_theorem_defect_m"] < 1e-12
    assert len(result["tables"]["meshes"]) == 3
