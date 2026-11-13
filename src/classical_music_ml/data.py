"""Load the competition files and clean them (section 3 of the notebook)."""

from pathlib import Path

import pandas as pd

from classical_music_ml.config import REPO_ROOT

# Folder with the competition csv files
DATA_DIR = REPO_ROOT / "data"

# Files the pipeline reads (concerts.csv and concerts_2014-15.csv are only used in the notebook)
REQUIRED_FILES = (
    "train.csv",
    "test.csv",
    "sample_submission.csv",
    "account.csv",
    "subscriptions.csv",
    "tickets_all.csv",
    "zipcodes.csv",
)

# The season we want to predict is 2014-15, so all the history we use must be from 2013-14 or before
TARGET_YEAR = 2014


def load_data(data_dir=DATA_DIR):
    """Reads all the files and returns a dictionary with the cleaned tables."""
    data_dir = Path(data_dir)
    missing = [name for name in REQUIRED_FILES if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing competition files in {}: {}. Download them from the Kaggle "
            "competition page (see data/README.md).".format(data_dir, ", ".join(missing))
        )

    # First we read the files (account.csv has some special characters, so we read it with latin-1)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample_submission = pd.read_csv(data_dir / "sample_submission.csv")
    accounts = pd.read_csv(data_dir / "account.csv", encoding="latin-1")
    subs = pd.read_csv(data_dir / "subscriptions.csv")
    tickets = pd.read_csv(data_dir / "tickets_all.csv")
    zipcodes = pd.read_csv(data_dir / "zipcodes.csv")

    # Then we create a numeric year for each season (2013-2014 -> 2013)
    subs["year"] = subs["season"].str[:4].astype(int)
    tickets["year"] = tickets["season"].str[:4].astype(int)

    # Only history from before the season we predict can be used. The competition files already stop
    # at 2013-14, so this changes nothing today; it guarantees the features can never see the label season
    subs = subs[subs["year"] < TARGET_YEAR].copy()
    tickets = tickets[tickets["year"] < TARGET_YEAR].copy()

    # Full subscriptions include all the concerts of the season, the other packages only a part of them
    subs["is_full"] = (subs["package"] == "Full").astype(int)

    # Ticket price levels mix numbers and text, so text values like 'Adult' or 'GA' become NaN
    tickets["price.level"] = pd.to_numeric(tickets["price.level"], errors="coerce")
    accounts["first.donated"] = pd.to_datetime(accounts["first.donated"])

    # We use the billing zip (or the shipping zip if billing is missing) and keep 5 characters
    zip5 = accounts["billing.zip.code"].fillna(accounts["shipping.zip.code"]).astype(str).str.strip().str[:5]

    # We add back the first zero when it was lost, and anything that is not 5 digits (foreign codes) becomes NaN
    zip5 = zip5.where(~zip5.str.fullmatch(r"\d{4}"), "0" + zip5)
    accounts["zip5"] = zip5.where(zip5.str.fullmatch(r"\d{5}"))

    # Finally we prepare the zipcodes table so we can join it with the accounts
    zipcodes["zip5"] = zipcodes["Zipcode"].astype(str).str.zfill(5)
    zipcodes["avg_income"] = zipcodes["TotalWages"] / zipcodes["TaxReturnsFiled"]

    return {
        "train": train,
        "test": test,
        "sample_submission": sample_submission,
        "accounts": accounts,
        "subs": subs,
        "tickets": tickets,
        "zipcodes": zipcodes,
    }
