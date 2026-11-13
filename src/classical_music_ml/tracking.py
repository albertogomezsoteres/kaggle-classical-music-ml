"""Run metadata: what was trained, on which data, with which code, and how it scored.

Each run writes ``runs/<timestamp>/metadata.json``. The competition data is not
in the repo, so the SHA-256 of every input file is recorded instead: two runs
with the same hashes, config and commit used exactly the same inputs.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from classical_music_ml.config import REPO_ROOT, Config
from classical_music_ml.data import REQUIRED_FILES

PACKAGES = ("classical-music-ml", "numpy", "pandas", "scikit-learn")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True)
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def package_versions() -> dict[str, str]:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def write_run_metadata(config: Config, metrics: dict[str, Any], outputs: dict[str, Path]) -> Path:
    """Write the metadata of one run and return the path of the JSON file."""
    started = datetime.now(timezone.utc)
    run_dir = config.runs_dir / started.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "timestamp_utc": started.isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "packages": package_versions(),
        "config": config.to_dict(),
        "inputs_sha256": {name: sha256(config.data_dir / name) for name in REQUIRED_FILES},
        "metrics": metrics,
        "outputs_sha256": {name: sha256(path) for name, path in outputs.items()},
    }
    path = run_dir / "metadata.json"
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return path
