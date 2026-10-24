import numpy as np


def canonical(value):
    import json

    from .results import native

    return json.dumps(native(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    import hashlib

    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def provenance(configuration):
    import platform
    import subprocess
    from importlib.metadata import version
    from pathlib import Path

    from . import __version__

    root = Path(__file__).resolve().parents[2]
    revision, dirty = None, None
    if (root / "src" / "emlab" / "__init__.py").is_file() and (root / "pyproject.toml").is_file():
        try:
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=True
            ).stdout.strip()
            dirty = bool(
                subprocess.run(
                    ["git", "status", "--porcelain"], cwd=root, text=True, capture_output=True, check=True
                ).stdout.strip()
            )
        except (OSError, subprocess.CalledProcessError):
            pass
    return {
        "package_version": __version__,
        "git_revision": revision,
        "git_dirty": dirty,
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "matplotlib": version("matplotlib"),
        "configuration_sha256": digest(configuration),
        "phasor_convention": "exp(+i omega t); outgoing exp(-i k r); SI units",
    }


def stamp(result, configuration, source=None):
    from .results import native, validate

    source = provenance(configuration) if source is None else dict(source)
    if source["configuration_sha256"] != digest(configuration):
        raise ValueError("source configuration hash mismatch")
    parameter_hash = digest(result["parameters"])
    if "parameters_sha256" in source and source["parameters_sha256"] != parameter_hash:
        raise ValueError("source parameter hash mismatch")
    source["parameters_sha256"] = parameter_hash
    return validate({**result, "configuration": native(configuration), "provenance": source})


def _array_hash(array):
    import hashlib

    array = np.asarray(array)
    header = canonical({"dtype": array.dtype.str, "shape": array.shape}).encode()
    return hashlib.sha256(header + array.tobytes(order="C")).hexdigest()


def _check_provenance(result):
    source = result["provenance"]
    if source["configuration_sha256"] != digest(result["configuration"]):
        raise ValueError("archived configuration hash mismatch")
    if source["parameters_sha256"] != digest(result["parameters"]):
        raise ValueError("archived physical parameter hash mismatch")


def save_archive(result, path):
    import os
    import tempfile
    from pathlib import Path

    from .results import native, validate

    validate(result)
    if "provenance" not in result:
        result = stamp(result, {"case": result["name"], "parameters": result["parameters"]})
    _check_provenance(result)
    arrays = {key: np.asarray(value) for key, value in result["arrays"].items()}
    metadata = native({key: value for key, value in result.items() if key != "arrays"})
    metadata["schema_version"] = 1
    metadata["array_sha256"] = {key: _array_hash(value) for key, value in arrays.items()}
    metadata["metadata_sha256"] = digest(metadata)
    arrays["__metadata__"] = np.array(canonical(metadata))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
            temporary = stream.name
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary is not None and Path(temporary).exists():
            Path(temporary).unlink()
    return path


def load_archive(path):
    import json

    from .results import validate

    with np.load(path, allow_pickle=False) as archive:
        if "__metadata__" not in archive.files:
            raise ValueError("missing field archive metadata")
        metadata = json.loads(str(archive["__metadata__"]))
        expected = metadata.pop("metadata_sha256", None)
        if expected != digest(metadata):
            raise ValueError("field archive metadata integrity mismatch")
        if metadata.pop("schema_version", None) != 1:
            raise ValueError("unsupported field archive schema")
        hashes = metadata.pop("array_sha256")
        if set(archive.files) != {*hashes, "__metadata__"}:
            raise ValueError("archive array inventory mismatch")
        arrays = {key: archive[key].copy() for key in hashes}
    for key, array in arrays.items():
        if _array_hash(array) != hashes[key]:
            raise ValueError("field archive array integrity mismatch: " + key)
    result = {**metadata, "arrays": arrays}
    _check_provenance(result)
    return validate(result)


def write_json(path, value):
    import json
    from pathlib import Path

    from .results import native

    Path(path).write_text(
        json.dumps(native(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def export_data(result, directory):
    import csv
    from pathlib import Path

    from .results import validate

    validate(result)
    if "provenance" not in result:
        result = stamp(result, {"case": result["name"], "parameters": result["parameters"]})
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("summary", result["summary"]),
        ("parameters", result["parameters"]),
        ("configuration", result["configuration"]),
        ("provenance", result["provenance"]),
    ):
        write_json(directory / (name + ".json"), value)
    tables = {"summary": [{"metric": key, "value": value} for key, value in result["summary"].items()]}
    tables.update({"table_" + key: value for key, value in result["tables"].items() if value})
    for name, rows in tables.items():
        if not rows:
            continue
        with (directory / (name + ".csv")).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    return directory
