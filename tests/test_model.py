"""The ensemble and its cross-validation."""

import numpy as np
import pytest

from classical_music_ml.config import CVConfig, ModelConfig
from classical_music_ml.features import build_final_features
from classical_music_ml.model import FinalEnsemble, cv_auroc

FAST = ModelConfig(rf_n_estimators=50)


@pytest.fixture(scope="module")
def xy(data: dict):
    return build_final_features(data["train"]["account.id"], data), data["train"]["label"].values


def test_probabilities_are_a_weighted_average_of_both_models(xy) -> None:
    X, y = xy
    model = FinalEnsemble(FAST).fit(X, y)
    p = model.predict_proba(X)
    expected = 0.7 * model.lr.predict_proba(X)[:, 1] + 0.3 * model.rf.predict_proba(X)[:, 1]
    np.testing.assert_allclose(p, expected)
    assert ((p >= 0) & (p <= 1)).all()


def test_training_is_deterministic(xy) -> None:
    X, y = xy
    first = FinalEnsemble(FAST).fit(X, y).predict_proba(X)
    second = FinalEnsemble(FAST).fit(X, y).predict_proba(X)
    np.testing.assert_allclose(first, second, atol=1e-12)


def test_cv_returns_one_score_per_fold_and_learns_the_signal(xy) -> None:
    X, y = xy
    aucs = cv_auroc(X, y, CVConfig(n_splits=5, n_repeats=3), FAST)
    assert aucs.shape == (15,)
    assert aucs.mean() > 0.7  # the synthetic label depends on loyalty, so the model must beat chance


def test_cv_folds_are_reproducible(xy) -> None:
    X, y = xy
    cv = CVConfig(n_splits=3, n_repeats=1)
    np.testing.assert_allclose(cv_auroc(X, y, cv, FAST), cv_auroc(X, y, cv, FAST), atol=1e-12)


@pytest.mark.parametrize(("lr_weight", "rf_weight"), [(0.7, 0.4), (1.2, -0.2)])
def test_invalid_ensemble_weights_are_rejected(lr_weight: float, rf_weight: float) -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        ModelConfig(lr_weight=lr_weight, rf_weight=rf_weight)
