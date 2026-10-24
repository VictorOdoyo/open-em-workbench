"""Publish a complete default catalog without changing its solver provenance."""

import argparse
import json
import shutil
from pathlib import Path

from emlab.batch import load_catalog, verify
from emlab.experiments import CASES, ExperimentConfig
from emlab.reporting import write_json


def build(source, destination):
    source, destination = Path(source), Path(destination)
    results = load_catalog(source)
    if {r["name"] for r in results} != set(CASES):
        raise ValueError("published gallery requires the complete current catalog")
    revisions = set()
    for result in results:
        if result["configuration"] != ExperimentConfig(result["name"]).to_dict():
            raise ValueError("published gallery requires default study requests")
        provenance = result["provenance"]
        if provenance["git_dirty"] is not False or not provenance["git_revision"]:
            raise ValueError("published gallery requires clean committed source captures")
        revisions.add(provenance["git_revision"])
    if len(revisions) != 1:
        raise ValueError("published gallery requires one source revision")
    verify(source)
    destination.mkdir(parents=True, exist_ok=True)
    for result in results:
        name = result["name"]
        manifest = json.loads((source / name / "manifest.json").read_text(encoding="utf-8"))
        target = destination / name
        target.mkdir(exist_ok=True)
        for filename in [*manifest, "manifest.json"]:
            shutil.copyfile(source / name / filename, target / filename)
    shutil.copyfile(source / "catalog.json", destination / "catalog.json")
    revision = next(iter(revisions))
    lines = [
        "# Measured experiment gallery",
        "",
        "Twenty-four default studies computed from clean source revision "
        + "[`"
        + revision[:12]
        + "`](https://github.com/VictorOdoyo/open-em-workbench/commit/"
        + revision
        + ").",
        "All values below come from the sealed captures. Each case directory contains the complete",
        "NPZ arrays, CSV tables, parameters, provenance, plot and HTML report.",
        "",
        "HTML files can be opened locally; GitHub previews the figures and tables directly.",
        "",
        "| Study | Selected measured outputs | Artifacts |",
        "| --- | --- | --- |",
    ]
    for result in results:
        name = result["name"]
        selected = []
        for key, value in list(result["summary"].items())[:2]:
            shown = (
                f"{value:.6g}"
                if isinstance(value, (float, int)) and not isinstance(value, bool)
                else str(value)
            )
            selected.append(key + " = " + shown)
        base = destination.name + "/" + name
        lines.append(
            "| ["
            + name
            + "]("
            + base
            + "/figure.png) | "
            + "; ".join(selected)
            + " | [summary]("
            + base
            + "/summary.csv), [setup]("
            + base
            + "/parameters.json), "
            + "[archive]("
            + base
            + "/fields.npz), [report]("
            + base
            + "/index.html) |"
        )
    lines += ["", "## Selected fields and diagnostics", ""]
    for name in (
        "fdtd-slab",
        "cylinder-scattering",
        "capacitance-coupling",
        "coil-mutual",
        "dielectric-design",
    ):
        lines += [
            "### " + name,
            "",
            "![" + CASES[name][2] + "](" + destination.name + "/" + name + "/figure.png)",
            "",
        ]
    lines += [
        "## Interpretation",
        "",
        "Use each report's assumptions before interpreting the figures. Small algebraic residuals,",
        "conservation defects and continuum errors are different diagnostics. Curved raster boundaries",
        "can show nonmonotonic mesh error. PML targets are not measured reflection guarantees.",
        "These outputs are simulations, not hardware measurements.",
        "",
        "Reproduce with `emlab batch --data-only --output outputs/catalog`, then",
        "`emlab render outputs/catalog` and `emlab verify outputs/catalog`.",
        "Replay an individual archive with `emlab replay ARCHIVE --output DIRECTORY`.",
        "Replay retains this source revision rather than claiming a fresh solve.",
        "",
    ]
    (destination.parent / "results.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    write_json(
        destination / "publication.json",
        {
            "source_revision": revision,
            "case_count": len(results),
            "request_defaults": {"resolution_scale": 1, "seed": 42},
        },
    )
    return len(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--destination", type=Path, default=Path("docs/gallery"))
    args = parser.parse_args()
    print(f"Published {build(args.source, args.destination)} verified study captures")
