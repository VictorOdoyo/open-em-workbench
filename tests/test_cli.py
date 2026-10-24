import pytest

from emlab.cli import main


def test_cli_lists_the_actual_catalog(capsys):
    assert main(["list"]) == 0
    output = capsys.readouterr().out
    assert "fdtd-debye" in output and "cylinder-scattering" in output


def test_cli_runs_physics_and_writes_report(tmp_path):
    from emlab.reporting import verify_report

    assert main(["run", "--case", "multilayer-coating", "--output", str(tmp_path)]) == 0
    assert verify_report(tmp_path)


def test_cli_rejects_config_override_ambiguity(tmp_path):
    path = tmp_path / "request.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        main(["run", "--config", str(path), "--scale", "2", "--output", str(tmp_path)])
    assert error.value.code == 2
