"""Solve an embedded electrode without using a built-in study."""

from pathlib import Path

from emlab.capacitance import extract
from emlab.geometry import conductor_labels, rectangle
from emlab.grid import from_bounds
from emlab.reporting import write_report
from emlab.results import image, make


def calculate():
    grid = from_bounds((-0.02, 0.02), (-0.015, 0.015), 81, 61)
    electrode = rectangle(grid, (-0.002, 0.002), (-0.006, 0.006))
    labels = conductor_labels(grid, [electrode])
    extracted = extract(grid, labels, epsilon_r=2.5)
    return make(
        "custom-capacitor",
        {"domain_m": [0.04, 0.03], "epsilon_r": 2.5, "voltage_v": 1},
        {"capacitance_f_m": float(extracted["matrix"][1, 1])},
        {"potential": extracted["basis"][1]},
        [image("One-volt electrode potential", "potential", grid, "Potential (V)")],
        assumptions=["Finite outer grounded conductor; capacitance is per out-of-plane metre."],
    )


if __name__ == "__main__":
    print(write_report(calculate(), Path("outputs/custom-capacitor")))
