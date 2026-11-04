# Trains the final model and writes the submission file
# Run it from the root of the repo:
#   python -m src.train         -> writes submissions/final_ensemble.csv
#   python -m src.train --cv    -> also prints the CV AUROC

import argparse
from pathlib import Path

import pandas as pd

from src.data import load_data
from src.features import build_final_features
from src.model import FinalEnsemble, cv_auroc

OUTPUT_PATH = Path(__file__).resolve().parents[1] / 'submissions' / 'final_ensemble.csv'


def main(run_cv=False):
    data = load_data()
    train, test = data['train'], data['test']

    # First we build the same 38 features for train and test
    X = build_final_features(train['account.id'], data)
    y = train['label'].values
    X_test = build_final_features(test['ID'], data)
    print('Train features: {} | Test features: {}'.format(X.shape, X_test.shape))

    # The CV is optional because it trains the model 15 times
    if run_cv:
        aucs = cv_auroc(X, y)
        print('CV AUROC: {:.4f} +/- {:.4f} ({} folds)'.format(aucs.mean(), aucs.std(), len(aucs)))

    # Then we train on all the training data and predict the test accounts
    model = FinalEnsemble().fit(X, y)
    submission = pd.DataFrame({'ID': test['ID'], 'Predicted': model.predict_proba(X_test)})

    # We check that the file has the same format as the sample submission
    sample = data['sample_submission']
    assert list(submission.columns) == list(sample.columns)
    assert (submission['ID'].values == sample['ID'].values).all()
    assert submission['Predicted'].between(0, 1).all()

    # Finally we save it
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    submission.to_csv(OUTPUT_PATH, index=False)
    print('Saved {} ({} rows, mean probability {:.4f})'.format(OUTPUT_PATH.name, len(submission),
                                                                submission['Predicted'].mean()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the final model and write the submission')
    parser.add_argument('--cv', action='store_true', help='also print the CV AUROC')
    main(run_cv=parser.parse_args().cv)
