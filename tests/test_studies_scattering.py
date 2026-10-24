from emlab import studies_scattering as m


def test_line_source_study_matches_complex_green_function_in_the_open_interior():
    summary = m.line_source()["summary"]
    assert summary["complex_field_relative_l2"] < 0.015
    assert summary["solver_residual"] < 1e-10
