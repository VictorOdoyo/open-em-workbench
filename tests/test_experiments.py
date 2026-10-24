import pytest

from emlab.experiments import CASES, ExperimentConfig, load_config


@pytest.mark.parametrize(
    "value",
    [
        {"case": "unknown"},
        {"resolution_scale": True},
        {"resolution_scale": 1.5},
        {"resolution_scale": 0},
        {"resolution_scale": 4},
        {"seed": -1},
        {"seed": True},
        {"seed": 2**32},
        {"extra": 1},
        [],
        None,
    ],
)
def test_request_rejects_ambiguous_or_unbounded_input(value):
    with pytest.raises(ValueError):
        ExperimentConfig.from_dict(value)


def test_request_json_roundtrip(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"case":"multilayer-coating","resolution_scale":2,"seed":7}', encoding="utf-8")
    config = load_config(path)
    assert ExperimentConfig.from_dict(config.to_dict()) == config
    assert len(CASES) == 24
