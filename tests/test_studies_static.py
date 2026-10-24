import pytest

from emlab import studies_static as m


def test_poisson_study_demonstrates_second_order_field_convergence():
    result = m.poisson()
    assert 1.95 < result["summary"]["observed_order"] < 2.05
    assert result["summary"]["finest_relative_error"] < 0.001
    assert len(result["tables"]["meshes"]) == 3


def test_interface_study_agrees_in_flux_charge_and_energy():
    summary = m.dielectric_interface()["summary"]
    assert summary["relative_displacement_variation"] < 1e-11
    assert summary["charge_capacitance_f_m"] == pytest.approx(
        summary["discrete_series_reference_f_m"], rel=1e-11
    )
    assert summary["energy_capacitance_f_m"] == pytest.approx(summary["charge_capacitance_f_m"], rel=1e-11)
