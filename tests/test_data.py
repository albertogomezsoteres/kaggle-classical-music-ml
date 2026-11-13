"""Loading and cleaning the competition files."""

from pathlib import Path

import pandas as pd
import pytest

from classical_music_ml.data import REQUIRED_FILES, load_data


def test_all_tables_are_loaded(data: dict) -> None:
    assert set(data) == {"train", "test", "sample_submission", "accounts", "subs", "tickets", "zipcodes"}


def test_seasons_become_numeric_years(data: dict) -> None:
    assert pd.api.types.is_integer_dtype(data["subs"]["year"])
    row = data["subs"].iloc[0]
    assert row["year"] == int(row["season"][:4])


def test_text_price_levels_become_missing(data: dict) -> None:
    prices = data["tickets"]["price.level"]
    assert pd.api.types.is_float_dtype(prices)
    assert prices.isna().any()  # 'Adult' and 'GA' in the raw file


def test_zip_codes_are_normalised_to_five_digits(data: dict) -> None:
    zip5 = data["accounts"]["zip5"].dropna()
    assert zip5.str.fullmatch(r"\d{5}").all()
    raw = data["accounts"]["billing.zip.code"].astype(str)
    lost_zero = raw.str.fullmatch(r"\d{4}")
    assert lost_zero.any()
    assert (data["accounts"].loc[lost_zero, "zip5"] == "0" + raw[lost_zero]).all()


def test_foreign_zip_codes_are_dropped(data: dict) -> None:
    foreign = data["accounts"]["billing.zip.code"] == "SW1A 1AA"
    assert foreign.any()
    assert data["accounts"].loc[foreign & data["accounts"]["shipping.zip.code"].isna(), "zip5"].isna().all()


def test_missing_files_are_named(tmp_path: Path) -> None:
    (tmp_path / "train.csv").write_text("account.id,label\n")
    with pytest.raises(FileNotFoundError) as err:
        load_data(tmp_path)
    for name in REQUIRED_FILES[1:]:
        assert name in str(err.value)
    assert "data/README.md" in str(err.value)
