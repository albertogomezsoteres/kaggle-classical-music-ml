"""Final model (weighted average of a logistic regression and a random forest) and its validation."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from classical_music_ml.config import CVConfig, ModelConfig


class FinalEnsemble:
    """Weighted average of a logistic regression and a random forest (70/30 in the winning model)."""

    def __init__(self, config: ModelConfig = ModelConfig()):
        self.config = config

        # The logistic regression needs standardized features, the random forest does not
        self.lr = make_pipeline(StandardScaler(), LogisticRegression(C=config.lr_C, max_iter=config.lr_max_iter))

        # min_samples_leaf=10 and max_features='sqrt' were the best values in the tuning
        self.rf = RandomForestClassifier(
            n_estimators=config.rf_n_estimators,
            min_samples_leaf=config.rf_min_samples_leaf,
            max_features=config.rf_max_features,
            n_jobs=-1,
            random_state=config.random_state,
        )

    def fit(self, X, y):
        """Fit both models on the same features."""
        self.lr.fit(X, y)
        self.rf.fit(X, y)
        return self

    def predict_proba(self, X):
        """Probability of buying a subscription (positive class only)."""
        c = self.config
        return c.lr_weight * self.lr.predict_proba(X)[:, 1] + c.rf_weight * self.rf.predict_proba(X)[:, 1]


def cv_auroc(X, y, cv: CVConfig = CVConfig(), model_config: ModelConfig = ModelConfig()):
    """Repeated stratified K-fold CV. Returns the AUROC of each fold (5 folds x 3 repeats = 15 values)."""
    # Stratified keeps the same proportion of buyers (4.8%) in every fold
    splitter = RepeatedStratifiedKFold(n_splits=cv.n_splits, n_repeats=cv.n_repeats, random_state=cv.random_state)

    aucs = []
    for train_index, val_index in splitter.split(X, y):
        # First we fit on the training folds, then we score the validation fold
        model = FinalEnsemble(model_config).fit(X.iloc[train_index], y[train_index])
        aucs.append(roc_auc_score(y[val_index], model.predict_proba(X.iloc[val_index])))

    return np.array(aucs)
