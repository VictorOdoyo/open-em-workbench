from emlab import studies_static as m


def test_poisson_study_demonstrates_second_order_field_convergence():
    result = m.poisson()
    assert 1.95 < result["summary"]["observed_order"] < 2.05
    assert result["summary"]["finest_relative_error"] < 0.001
    assert len(result["tables"]["meshes"]) == 3
