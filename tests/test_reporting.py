import numpy as np

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


def _example():
    from emlab.results import curve, make

    return make(
        "sample",
        {"frequency_hz": 1e9},
        {"relative_error": 0.01},
        {"x": [1.0, 2.0, 3], "y": [0.1, 0.02, 0.01], "field": [1 + 2j, 3 + 4j]},
        [curve("Convergence", "x", [("y", "measured")], "h (m)", "Relative error")],
        {"samples": [{"x_m": 1.0, "error": 0.1}, {"x_m": 2.0, "error": 0.02}]},
    )


def test_archive_contains_numeric_fields_and_unicode_metadata_without_pickle(tmp_path):
    path = m.save_archive(_example(), tmp_path / "fields.npz")
    with np.load(path, allow_pickle=False) as archive:
        assert all(archive[key].dtype.kind != "O" for key in archive.files)
        assert np.iscomplexobj(archive["field"])
        assert archive["__metadata__"].dtype.kind == "U"
