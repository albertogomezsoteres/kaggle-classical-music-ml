"""Train the final model and write the submission file (single entry point).

classical-music-ml                 # writes submissions/final_ensemble.csv
classical-music-ml --cv            # also prints the 15-fold CV AUROC
classical-music-ml --config other.yaml --data-dir /path/to/csvs
"""

import argparse

import pandas as pd

from classical_music_ml.config import DEFAULT_CONFIG, Config, load_config
from classical_music_ml.data import load_data
from classical_music_ml.features import build_final_features
from classical_music_ml.model import FinalEnsemble, cv_auroc
from classical_music_ml.tracking import write_run_metadata


def train(config: Config, run_cv: bool = False) -> pd.DataFrame:
    """Build the features, optionally cross-validate, fit on all the training data and write the submission."""
    data = load_data(config.data_dir)
    train_df, test_df = data["train"], data["test"]

    # First we build the same 38 features for train and test
    X = build_final_features(train_df["account.id"], data)
    y = train_df["label"].values
    X_test = build_final_features(test_df["ID"], data)
    print("Train features: {} | Test features: {}".format(X.shape, X_test.shape))

    # The CV is optional because it trains the model 15 times
    metrics = {"n_train": len(y), "n_test": len(X_test), "positive_rate": float(y.mean())}
    if run_cv:
        aucs = cv_auroc(X, y, config.cv, config.model)
        print("CV AUROC: {:.4f} +/- {:.4f} ({} folds)".format(aucs.mean(), aucs.std(), len(aucs)))
        metrics.update(cv_auroc_mean=float(aucs.mean()), cv_auroc_std=float(aucs.std()), cv_auroc_folds=aucs.tolist())

    # Then we train on all the training data and predict the test accounts
    model = FinalEnsemble(config.model).fit(X, y)
    submission = pd.DataFrame({"ID": test_df["ID"], "Predicted": model.predict_proba(X_test)})

    # We check that the file has the same format as the sample submission
    sample = data["sample_submission"]
    assert list(submission.columns) == list(sample.columns)
    assert (submission["ID"].values == sample["ID"].values).all()
    assert submission["Predicted"].between(0, 1).all()

    # Finally we save it, with the metadata of the run next to it
    config.submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(config.submission_path, index=False)
    print(
        "Saved {} ({} rows, mean probability {:.4f})".format(
            config.submission_path.name, len(submission), submission["Predicted"].mean()
        )
    )
    meta = write_run_metadata(config, metrics, {"submission": config.submission_path})
    print("Run metadata: {}".format(meta))
    return submission


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train the final model and write the submission")
    parser.add_argument("--cv", action="store_true", help="also print the CV AUROC")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML config (default: configs/final_model.yaml)")
    parser.add_argument("--data-dir", help="folder with the competition CSV files (overrides the config)")
    parser.add_argument("--output", help="submission path (overrides the config)")
    args = parser.parse_args(argv)
    config = load_config(args.config, data_dir=args.data_dir, submission_path=args.output)
    train(config, run_cv=args.cv)


if __name__ == "__main__":
    main()
