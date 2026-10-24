import numpy as np

from emlab import studies_magnetic as m


def test_coil_study_converges_to_elliptic_reference_and_has_attractive_force():
    result = m.coil_mutual()
    assert result["summary"]["finest_relative_error"] < 0.001
    assert result["summary"]["coaxial_attractive_force_n"] < 0
    assert np.any(result["arrays"]["excluded"])
