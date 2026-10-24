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
