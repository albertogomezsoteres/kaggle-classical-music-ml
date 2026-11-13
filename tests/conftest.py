"""Shared fixtures: a synthetic copy of the competition data and a config that points at it."""

from __future__ import annotations

from pathlib import Path

import pytest
from synthetic import write_synthetic_competition

from classical_music_ml.config import Config, load_config
from classical_music_ml.data import load_data


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_synthetic_competition(tmp_path_factory.mktemp("competition"))


@pytest.fixture(scope="session")
def data(data_dir: Path) -> dict:
    return load_data(data_dir)


@pytest.fixture
def config(data_dir: Path, tmp_path: Path) -> Config:
    """The real config file, redirected to the synthetic data and a temporary output folder."""
    return load_config(
        data_dir=data_dir,
        submission_path=tmp_path / "submission.csv",
        runs_dir=tmp_path / "runs",
    )
