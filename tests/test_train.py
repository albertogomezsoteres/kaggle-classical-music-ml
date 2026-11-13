"""End to end: config file -> features -> model -> submission + run metadata."""

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import yaml

from classical_music_ml.config import DEFAULT_CONFIG, ModelConfig, load_config
from classical_music_ml.train import main, train


def test_default_config_is_the_winning_model() -> None:
    config = load_config()
    assert config.model == ModelConfig()  # defaults and YAML agree: 70/30, C=1, 800 trees, leaf 10
    assert (config.cv.n_splits, config.cv.n_repeats) == (5, 3)
    assert config.submission_path.name == "final_ensemble.csv"


def test_train_writes_a_valid_submission_and_its_metadata(config, data_dir: Path) -> None:
    config = replace(config, model=replace(config.model, rf_n_estimators=50))
    submission = train(config, run_cv=True)

    written = pd.read_csv(config.submission_path)
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    assert list(written.columns) == ["ID", "Predicted"]
    assert (written["ID"] == sample["ID"]).all()
    assert written["Predicted"].between(0, 1).all()
    pd.testing.assert_frame_equal(written, submission.reset_index(drop=True), check_exact=False)

    (meta_path,) = config.runs_dir.glob("*/metadata.json")
    meta = json.loads(meta_path.read_text())
    assert len(meta["metrics"]["cv_auroc_folds"]) == 15
    assert set(meta["inputs_sha256"]) >= {"train.csv", "test.csv", "account.csv"}
    assert all(len(h) == 64 for h in meta["inputs_sha256"].values())
    assert meta["config"]["model"]["rf_n_estimators"] == 50
    assert "submission" in meta["outputs_sha256"]


def test_command_line_entry_point(data_dir: Path, tmp_path: Path) -> None:
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    raw["model"]["rf_n_estimators"] = 50
    raw["runs_dir"] = str(tmp_path / "runs")
    cfg = tmp_path / "fast.yaml"
    cfg.write_text(yaml.safe_dump(raw))

    out = tmp_path / "cli_submission.csv"
    main(["--config", str(cfg), "--data-dir", str(data_dir), "--output", str(out)])
    assert len(pd.read_csv(out)) == len(pd.read_csv(data_dir / "test.csv"))
