import numpy as np

from emlab import studies_time as m


def test_broadband_yee_slab_agrees_with_independent_multilayer_complex_amplitudes():
    result = m.slab()
    summary = result["summary"]
    assert np.max(np.diff(result["arrays"]["snapshot_times_ns"])) <= 1e9 / (8 * 3e9) * (1 + 1e-9)
    assert summary["valid_frequency_points"] == 201
    assert summary["reflection_complex_relative_l2"] < 0.03
    assert summary["transmission_complex_relative_l2"] < 0.03
    assert summary["maximum_absorption_error"] < 0.03


def test_dispersive_time_domain_slab_is_passive_and_matches_causal_frequency_response():
    summary = m.debye_slab()["summary"]
    assert summary["reflection_complex_relative_l2"] < 0.03
    assert summary["transmission_complex_relative_l2"] < 0.03
    assert summary["maximum_absorption_error"] < 0.03
    assert summary["minimum_measured_absorption"] > 0


def test_cavity_study_matches_discrete_frequency_and_conserves_modified_energy():
    summary = m.cavity()["summary"]
    assert summary["relative_discrete_frequency_error"] < 1e-9
    assert summary["maximum_relative_discrete_energy_drift"] < 1e-11
    assert summary["continuum_frequency_relative_error"] < 0.003


def test_lossy_cavity_study_recovers_positive_quality_factor_and_discrete_decay():
    summary = m.lossy_cavity()["summary"]
    assert summary["relative_discrete_frequency_error"] < 1e-9
    assert summary["relative_discrete_quality_error"] < 1e-8
    assert 1 < summary["quality_factor_measured"] < 100
    assert summary["pole_fit_residual"] < 1e-10


def test_absorber_study_retains_large_pec_return_and_reports_measured_cpml_limit():
    summary = m.absorber()["summary"]
    assert summary["pec_return_amplitude"] > 0.8
    assert summary["cpml_return_amplitude"] < 0.002
    assert summary["mur_return_amplitude"] < 0.01
