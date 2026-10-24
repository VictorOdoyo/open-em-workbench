from emlab import studies_time as m


def test_broadband_yee_slab_agrees_with_independent_multilayer_complex_amplitudes():
    summary = m.slab()["summary"]
    assert summary["valid_frequency_points"] == 201
    assert summary["reflection_complex_relative_l2"] < 0.03
    assert summary["transmission_complex_relative_l2"] < 0.03
    assert summary["maximum_absorption_error"] < 0.03
