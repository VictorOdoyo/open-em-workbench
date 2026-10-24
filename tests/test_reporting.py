from emlab import reporting as m


def test_provenance_hashes_actual_requests_and_preserves_frozen_source():
    from emlab.results import curve, make

    config = {"case": "sample", "resolution_scale": 1, "seed": 42}
    frozen = m.provenance(config)
    result = make(
        "sample",
        {"epsilon_r": 3},
        {"error": 0.01},
        {"x": [1.0, 2], "y": [0.1, 0.01]},
        [curve("Error", "x", [("y", "measured")], "h (m)", "Relative error")],
    )
    frozen["git_revision"] = "original-source"
    result = m.stamp(result, config, frozen)
    assert result["provenance"]["git_revision"] == "original-source"
    assert result["provenance"]["parameters_sha256"] == m.digest({"epsilon_r": 3})
    assert m.digest(config) != m.digest({**config, "resolution_scale": 2})
