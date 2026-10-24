from emlab import studies_time as m


def test_broadband_yee_slab_agrees_with_independent_multilayer_complex_amplitudes():
    summary = m.slab()["summary"]
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
