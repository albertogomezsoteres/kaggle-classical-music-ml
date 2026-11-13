"""The feature pipeline: same columns for train and test, no information from the target season."""

import numpy as np
import pandas as pd
import pytest
from synthetic import _account_id

from classical_music_ml.data import TARGET_YEAR, load_data
from classical_music_ml.features import build_features, build_final_features, id_number, id_rank, streak_until


@pytest.mark.parametrize(
    ("years", "last_year", "expected"),
    [([2011, 2012, 2013], 2013, 3), ([2010, 2012, 2013], 2013, 2), ([2010, 2011], 2013, 0), ([], 2013, 0)],
)
def test_streak_counts_consecutive_seasons_backwards(years, last_year, expected) -> None:
    assert streak_until(years, last_year) == expected


@pytest.mark.parametrize("counter", [0, 1, 61, 62, 3843, 10**9])
def test_id_number_decodes_the_base62_counter(counter: int) -> None:
    assert id_number(_account_id(counter)) == counter


def test_account_age_is_a_rank_between_0_and_1(data: dict) -> None:
    ids = data["train"]["account.id"]
    rank = id_rank(ids, data["accounts"])
    assert ((rank > 0) & (rank <= 1)).all()
    order = np.argsort([id_number(i) for i in ids])
    assert np.all(np.diff(rank[order]) >= 0)  # older IDs never rank after newer ones


def test_train_and_test_get_the_same_38_features(data: dict) -> None:
    X = build_final_features(data["train"]["account.id"], data)
    X_test = build_final_features(data["test"]["ID"], data)
    assert X.shape[1] == 38
    assert list(X.columns) == list(X_test.columns)
    assert build_features(data["train"]["account.id"], data).shape[1] == 37
    assert X.notna().all().all()
    assert X_test.notna().all().all()
    assert list(X.index) == list(data["train"]["account.id"])


def test_accounts_without_history_get_the_documented_defaults(data: dict) -> None:
    X = build_features(data["train"]["account.id"], data)
    no_history = X[(X["has_sub"] == 0) & (X["has_tix"] == 0)]
    assert len(no_history) > 0
    assert (no_history["years_since_sub"] == 30).all()
    assert (no_history["years_since_tix"] == 30).all()
    assert (no_history["sub_n_seasons"] == 0).all()
    assert (no_history["sub_last_price_level"] == 2.5).all()


def test_target_season_is_never_used(data_dir, tmp_path) -> None:
    """History rows from the 2014-15 season must not change a single feature (no label leakage)."""
    data = load_data(data_dir)
    ids = data["train"]["account.id"]
    before = build_final_features(ids, data)

    # Copy the files and append 2014-15 subscriptions and tickets for some accounts
    for f in data_dir.iterdir():
        (tmp_path / f.name).write_bytes(f.read_bytes())
    target = f"{TARGET_YEAR}-{TARGET_YEAR + 1}"
    for name in ("subscriptions.csv", "tickets_all.csv"):
        rows = pd.read_csv(data_dir / name)
        pd.concat([rows, rows.head(50).assign(season=target)]).to_csv(tmp_path / name, index=False)

    after = build_final_features(ids, load_data(tmp_path))
    pd.testing.assert_frame_equal(before, after)
