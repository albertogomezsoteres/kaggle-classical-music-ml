# Final model (weighted average of a logistic regression and a random forest) and its validation

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Weight of each model in the average. 70/30 was the best combination in the CV
LR_WEIGHT, RF_WEIGHT = 0.7, 0.3


class FinalEnsemble:

    def __init__(self):
        # The logistic regression needs standardized features, the random forest does not
        self.lr = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=5000))

        # min_samples_leaf=10 and max_features='sqrt' were the best values in the tuning
        self.rf = RandomForestClassifier(n_estimators=800, min_samples_leaf=10, max_features='sqrt',
                                         n_jobs=-1, random_state=0)

    def fit(self, X, y):
        self.lr.fit(X, y)
        self.rf.fit(X, y)
        return self

    def predict_proba(self, X):
        # We return only the probability of buying a subscription (positive class)
        return LR_WEIGHT * self.lr.predict_proba(X)[:, 1] + RF_WEIGHT * self.rf.predict_proba(X)[:, 1]


# Repeated stratified K-fold CV. Returns the AUROC of each fold (5 folds x 3 repeats = 15 values)
def cv_auroc(X, y, n_splits=5, n_repeats=3, random_state=0):
    # Stratified keeps the same proportion of buyers (4.8%) in every fold
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    aucs = []
    for train_index, val_index in cv.split(X, y):
        # First we fit on the training folds, then we score the validation fold
        model = FinalEnsemble().fit(X.iloc[train_index], y[train_index])
        aucs.append(roc_auc_score(y[val_index], model.predict_proba(X.iloc[val_index])))

    return np.array(aucs)
