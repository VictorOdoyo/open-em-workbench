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
