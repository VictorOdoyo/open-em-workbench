"""Validated reproducible run requests and lazy study dispatch."""

import json
from dataclasses import dataclass
from pathlib import Path

CASES = {
    "poisson-convergence": ("studies_static", "poisson", "Manufactured electrostatic mesh convergence"),
    "dielectric-interface": (
        "studies_static",
        "dielectric_interface",
        "Displacement continuity across dielectric interfaces",
    ),
    "capacitance-coupling": (
        "studies_static",
        "coupled_capacitors",
        "Maxwell capacitance matrix and floating conductor",
    ),
    "coax-convergence": (
        "studies_static",
        "coaxial_convergence",
        "Rasterized coaxial capacitance convergence",
    ),
    "multilayer-coating": ("studies_waves", "coating", "Quarter-wave antireflection coating"),
    "bragg-mirror": ("studies_waves", "bragg", "Bragg stop band and Bloch attenuation"),
    "absorbing-stack": ("studies_waves", "absorbing_stack", "Layer absorption and volume Joule power"),
    "microwave-lines": ("studies_waves", "microwave_lines", "Lossy coax and impedance matching"),
    "cavity-cutoffs": ("studies_modes", "cavity_cutoffs", "TE and TM cavity eigenmodes"),
    "slab-guidance": ("studies_modes", "slab_guidance", "Slab eigenmodes versus transcendental reference"),
    "mode-perturbation": ("studies_modes", "mode_perturbation", "Dielectric eigenfrequency perturbation"),
    "coil-mutual": ("studies_magnetic", "coil_mutual", "Neumann mutual inductance and coil field"),
    "magnetostatic-current": (
        "studies_magnetic",
        "current_fields",
        "Vector potential, Ampere circulation and energy",
    ),
    "fdtd-slab": ("studies_time", "slab", "Two-run time-domain dielectric slab scattering"),
    "fdtd-debye": ("studies_time", "debye_slab", "Dispersive Debye ADE slab scattering"),
    "fdtd-cavity": ("studies_time", "cavity", "Yee cavity ringdown and discrete energy invariant"),
    "fdtd-lossy-cavity": ("studies_time", "lossy_cavity", "Conductive cavity quality factor"),
    "fdtd-absorber": ("studies_time", "absorber", "Measured PEC, Mur and CPML pulse return"),
    "fdfd-line-source": (
        "studies_scattering",
        "line_source",
        "Outgoing source field versus Hankel Green function",
    ),
    "cylinder-scattering": (
        "studies_scattering",
        "cylinder",
        "PEC cylinder scattering versus harmonic series",
    ),
    "dielectric-sensitivity": (
        "studies_design",
        "sensitivity",
        "Exact discrete material gradient verification",
    ),
    "maxwell-force": ("studies_design", "force", "Stress contour force versus grounded image line"),
    "dielectric-design": ("studies_design", "design", "Material-budget constrained dielectric optimization"),
    "manufacturing-uncertainty": ("studies_design", "uncertainty", "Correlated coax tolerance propagation"),
}


@dataclass(frozen=True)
class ExperimentConfig:
    case: str = "poisson-convergence"
    resolution_scale: int = 1
    seed: int = 42

    def __post_init__(self):
        if not isinstance(self.case, str) or self.case not in CASES:
            raise ValueError("unknown experiment case")
        if type(self.resolution_scale) is not int or not 1 <= self.resolution_scale <= 3:
            raise ValueError("resolution_scale must be an integer from 1 to 3")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")

    def to_dict(self):
        return {"case": self.case, "resolution_scale": self.resolution_scale, "seed": self.seed}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) - {"case", "resolution_scale", "seed"}:
            raise ValueError("request must be an object containing only case, resolution_scale and seed")
        return cls(**value)


def load_config(path):
    return ExperimentConfig.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def run(config, progress=None):
    from importlib import import_module

    from .reporting import provenance, stamp
    from .results import validate

    if not isinstance(config, ExperimentConfig):
        raise ValueError("validated ExperimentConfig required")
    configuration = config.to_dict()
    source = provenance(configuration)
    module, function, _ = CASES[config.case]
    if progress is not None:
        progress("Solving " + config.case)
    result = getattr(import_module("emlab." + module), function)(
        scale=config.resolution_scale, seed=config.seed
    )
    validate(result)
    if result["name"] != config.case:
        raise ValueError("study returned a different case name")
    return stamp(result, configuration, source)
