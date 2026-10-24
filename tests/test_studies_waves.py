import pytest

from emlab import studies_waves as m


def test_antireflection_study_matches_at_design_and_conserves_power():
    summary = m.coating()["summary"]
    assert summary["design_reflectance"] < 1e-25
    assert summary["maximum_power_balance_defect"] < 1e-12


def test_bragg_study_has_a_stopband_without_fabricating_absorption():
    summary = m.bragg()["summary"]
    assert summary["design_reflectance"] > 0.999
    assert summary["design_transmittance"] < 0.001
    assert summary["maximum_lossless_balance_defect"] < 1e-11


def test_absorbing_stack_study_agrees_in_global_local_flux_and_joule_energy():
    summary = m.absorbing_stack()["summary"]
    assert 0 < summary["total_absorption"] < 1
    assert summary["total_absorption"] == pytest.approx(summary["layer_absorption_sum"], abs=1e-12)
    assert summary["maximum_joule_flux_defect"] < 2e-5
