import runpy
from pathlib import Path

import numpy as np


def test_custom_capacitor_example_uses_real_extraction():
    example = Path(__file__).resolve().parents[1] / "examples/custom_capacitor.py"
    result = runpy.run_path(str(example))["calculate"]()
    assert result["summary"]["capacitance_f_m"] > 0
    assert np.max(result["arrays"]["potential"]) == 1
