"""Typed configuration loaded from ``configs/final_model.yaml``."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "final_model.yaml"


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters of the logistic regression + random forest ensemble."""

    lr_weight: float = 0.7
    rf_weight: float = 0.3
    lr_C: float = 1.0
    lr_max_iter: int = 5000
    rf_n_estimators: int = 800
    rf_min_samples_leaf: int = 10
    rf_max_features: str = "sqrt"
    random_state: int = 0

    def __post_init__(self) -> None:
        if min(self.lr_weight, self.rf_weight) < 0 or abs(self.lr_weight + self.rf_weight - 1) > 1e-9:
            raise ValueError(
                f"ensemble weights must be non-negative and sum to 1, got {self.lr_weight} + {self.rf_weight}"
            )


@dataclass(frozen=True)
class CVConfig:
    """Repeated stratified K-fold settings, shared by every model comparison."""

    n_splits: int = 5
    n_repeats: int = 3
    random_state: int = 0


@dataclass(frozen=True)
class Config:
    """Paths and settings for one training run. Relative paths are resolved from the repo root."""

    data_dir: Path = REPO_ROOT / "data"
    submission_path: Path = REPO_ROOT / "submissions" / "final_ensemble.csv"
    runs_dir: Path = REPO_ROOT / "runs"
    model: ModelConfig = field(default_factory=ModelConfig)
    cv: CVConfig = field(default_factory=CVConfig)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        for key in ("data_dir", "submission_path", "runs_dir"):
            out[key] = str(out[key])
        return out


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: str | Path = DEFAULT_CONFIG, **overrides: Any) -> Config:
    """Read a YAML config; keyword overrides (e.g. ``data_dir=...``) take precedence."""
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    raw.update({k: v for k, v in overrides.items() if v is not None})
    return Config(
        data_dir=_resolve(raw.get("data_dir", "data")),
        submission_path=_resolve(raw.get("submission_path", "submissions/final_ensemble.csv")),
        runs_dir=_resolve(raw.get("runs_dir", "runs")),
        model=ModelConfig(**raw.get("model", {})),
        cv=CVConfig(**raw.get("cv", {})),
    )
