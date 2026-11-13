# Model card: orchestra subscription model

## Overview

| | |
|-|-|
| Task | Binary classification: will a past patron buy a subscription for the 2014-15 season? |
| Model | 70% logistic regression (C = 1, standardized features) + 30% random forest (800 trees, `min_samples_leaf` = 10) |
| Features | 38: subscription history (18), single tickets (8), donations (6), location (5), account age (1) |
| Metric | AUROC |
| Result | 0.98908 on the Kaggle private leaderboard (first place); 0.9865 ± 0.0051 in 15-fold cross-validation |
| Context | AIPI 520: Modeling Processes and Algorithms, Duke University, Fall 2026 |

## Intended use

Ranking past patrons by how likely they are to subscribe, so an orchestra can prioritize renewal and outreach
campaigns. The output is a score for ordering patrons, not a calibrated probability, and it was built for a
course competition, not for production.

## Training data

6,941 labelled accounts (332 buyers, 4.8%) and 2,975 test accounts, with subscription, ticket and donation
history up to the 2013-14 season and zip-code demographics. The data is not included in this repository.

## Evaluation

- Repeated stratified 5-fold cross-validation (3 repeats, 15 folds), the same splits for every model, so models
  are compared fold by fold.
- No model, feature or weight was chosen with the leaderboard.
- The final model beats the first submission in all 15 folds.

## Limitations

- **Account age** is derived from the order of the account IDs. It does not describe the patron, it comes from
  how the database creates IDs. It uses no information from 2014-15 and is computed the same way for train and
  test, but in a real deployment the account creation date should be confirmed to be known before the season.
- One orchestra and one season. The model has not been tested on other venues or later years.
- Small positive class (332 buyers): fold-to-fold standard deviation is about 0.005 AUROC.
- The tests run on synthetic data, which checks the code but not the model's accuracy.

## Ethical considerations

The features include donation amounts and zip-code income. A score like this is reasonable for deciding whom to
contact first; it should not be used to exclude patrons from offers or to set individual prices.
