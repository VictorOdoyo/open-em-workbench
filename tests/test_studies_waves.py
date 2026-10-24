from emlab import studies_waves as m


def test_antireflection_study_matches_at_design_and_conserves_power():
    summary = m.coating()["summary"]
    assert summary["design_reflectance"] < 1e-25
    assert summary["maximum_power_balance_defect"] < 1e-12
