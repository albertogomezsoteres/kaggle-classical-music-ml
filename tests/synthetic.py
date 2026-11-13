"""A small synthetic copy of the competition files, with the same schema.

The real data cannot be redistributed, so the tests run on this instead. It
keeps the quirks the pipeline has to handle: Salesforce-style account IDs,
zip codes that lost their leading zero or are foreign, mixed text and numeric
ticket price levels, latin-1 text in account.csv, and a rare positive class.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

B62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
SEASONS = [f"{y}-{y + 1}" for y in range(2000, 2015)]  # up to 2014-2015, the target season
LOCATIONS = ["San Francisco", "Berkeley", "Peninsula", "Family concert"]
STATES = ["CA", "CA", "CA", "NY", "OR"]


def _account_id(counter: int) -> str:
    """Salesforce-style 15-character ID whose characters 5-14 are a base-62 counter."""
    digits = []
    for _ in range(10):
        counter, r = divmod(counter, 62)
        digits.append(B62[r])
    return "001i0" + "".join(reversed(digits))


def _random_date(rng: np.random.Generator) -> str:
    return f"{rng.integers(1990, 2015)}-0{rng.integers(1, 9)}-1{rng.integers(0, 9)}"


def write_synthetic_competition(data_dir: Path, n_accounts: int = 400, n_test: int = 120, seed: int = 0) -> Path:
    """Write the nine competition CSV files to ``data_dir`` and return it."""
    rng = np.random.default_rng(seed)
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    counters = np.sort(rng.choice(10**8, size=n_accounts, replace=False)) + 10**6
    ids = [_account_id(int(c)) for c in counters]
    loyalty = rng.random(n_accounts)  # latent propensity that drives history and label

    # Zip codes: some lost their leading zero, some are foreign, some missing
    zip_pool = [f"{z:05d}" for z in rng.choice(np.arange(1000, 99999), size=60, replace=False)]
    billing = []
    for _ in range(n_accounts):
        roll = rng.random()
        if roll < 0.08:
            billing.append(None)
        elif roll < 0.12:
            billing.append("SW1A 1AA")  # foreign
        else:
            z = str(rng.choice(zip_pool))
            billing.append(z.lstrip("0") if rng.random() < 0.1 else z + ("-1234" if rng.random() < 0.2 else ""))
    accounts = pd.DataFrame(
        {
            "account.id": ids,
            "shipping.zip.code": [str(rng.choice(zip_pool)) if rng.random() < 0.5 else None for _ in ids],
            "billing.zip.code": billing,
            "billing.city": ["Ciudad Pública" if rng.random() < 0.05 else "Oakland" for _ in ids],
            "relationship": [("Board" if rng.random() < 0.1 else None) for _ in ids],
            "amount.donated.2013": np.round(rng.exponential(200, n_accounts) * (rng.random(n_accounts) < 0.3), 2),
            "amount.donated.lifetime": np.round(rng.exponential(1500, n_accounts) * (rng.random(n_accounts) < 0.5), 2),
            "no.donations.lifetime": rng.poisson(2, n_accounts),
            "first.donated": [
                (
                    f"{rng.integers(1990, 2015)}-0{rng.integers(1, 9)}-1{rng.integers(0, 9)}"
                    if rng.random() < 0.5
                    else None
                )
                for _ in ids
            ],
        }
    )
    accounts.to_csv(data_dir / "account.csv", index=False, encoding="latin-1")

    subs, tickets = [], []
    for i, acc in enumerate(ids):
        n_sub_seasons = int(loyalty[i] * 12 * (rng.random() < 0.6))
        start = rng.integers(0, len(SEASONS) - n_sub_seasons) if n_sub_seasons else 0
        # Recent loyal subscribers tend to keep subscribing up to 2013-14
        recent = loyalty[i] > 0.6
        seasons = SEASONS[-1 - n_sub_seasons : -1] if recent else SEASONS[start : start + n_sub_seasons]
        for season in seasons:
            subs.append(
                {
                    "account.id": acc,
                    "season": season,
                    "package": rng.choice(["Full", "Quartet", "Trio"]),
                    "no.seats": int(rng.integers(1, 5)),
                    "location": rng.choice(LOCATIONS[:3]),
                    "section": rng.choice(["Premium Orchestra", "Orchestra", "Balcony"]),
                    "price.level": float(rng.integers(1, 5)),
                    "subscription_tier": float(rng.integers(1, 4)),
                    "multiple.subs": rng.choice(["yes", "no"], p=[0.1, 0.9]),
                }
            )
        for _ in range(int(rng.poisson(0.6))):
            tickets.append(
                {
                    "account.id": acc,
                    "season": rng.choice(SEASONS[-6:-1]),
                    "location": rng.choice(LOCATIONS),
                    "set": int(rng.integers(1, 9)),
                    "no.seats": int(rng.integers(1, 5)),
                    "price.level": rng.choice(["1", "2", "3", "Adult", "GA"]),
                }
            )
    pd.DataFrame(subs).to_csv(data_dir / "subscriptions.csv", index=False)
    pd.DataFrame(tickets).to_csv(data_dir / "tickets_all.csv", index=False)

    zipcodes = pd.DataFrame(
        {
            "Zipcode": [int(z) for z in zip_pool],
            "ZipCodeType": "STANDARD",
            "City": "City",
            "State": [STATES[k % len(STATES)] for k in range(len(zip_pool))],
            "LocationType": "PRIMARY",
            "Lat": rng.uniform(32, 45, len(zip_pool)),
            "Long": rng.uniform(-124, -70, len(zip_pool)),
            "Location": "NA-US",
            "Decommisioned": False,
            "TaxReturnsFiled": rng.integers(100, 20000, len(zip_pool)),
            "EstimatedPopulation": rng.integers(500, 50000, len(zip_pool)),
            "TotalWages": rng.integers(10**6, 10**9, len(zip_pool)),
        }
    )
    zipcodes.to_csv(data_dir / "zipcodes.csv", index=False)

    concerts = pd.DataFrame(
        {
            "season": SEASONS[:-1],
            "concert.name": "Concert",
            "set": 1,
            "who": "Orchestra",
            "what": "Symphony",
            "location": "San Francisco",
        }
    )
    concerts.to_csv(data_dir / "concerts.csv", index=False)
    concerts.head(6).assign(season="2014-2015").to_csv(data_dir / "concerts_2014-15.csv", index=False)

    # Labels: a rare positive class (~5-15%) driven by loyalty
    order = rng.permutation(n_accounts)
    test_idx, train_idx = order[:n_test], order[n_test:]
    p = 1 / (1 + np.exp(-(14 * (loyalty[train_idx] - 0.85))))
    label = (rng.random(len(train_idx)) < p).astype(int)
    pd.DataFrame({"account.id": [ids[i] for i in train_idx], "label": label}).to_csv(
        data_dir / "train.csv", index=False
    )
    test_ids = [ids[i] for i in test_idx]
    pd.DataFrame({"ID": test_ids}).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"ID": test_ids, "Predicted": 0.5}).to_csv(data_dir / "sample_submission.csv", index=False)
    return data_dir
