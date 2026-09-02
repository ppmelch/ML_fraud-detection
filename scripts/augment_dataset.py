#!/usr/bin/env python3
"""
Generate synthetic transactions for the US states the real dataset never
touches, so the state choropleth has a value for every GeoJSON polygon.

The real ``data/bank_transactions_data_2.csv`` has 43 city names covering only
27 of the 52 GeoJSON features. The remaining 25 states render grey. This
script builds statistically-faithful synthetic rows for those 25 states by
**resampling the real dataset with replacement** and overriding only the
fields that must change (location, identifiers) plus a light multiplicative
jitter on the continuous columns. Real rows are copied through untouched;
synthetic rows are identifiable by their ``TXS`` / ``ACS`` id prefix.

The output is written to ``data/bank_transactions_data_2_full.csv`` with a
schema byte-identical to the source (no extra column). ``config.DATASET_PATH``
points at that file, so re-running ``scripts.train_pipeline`` afterwards makes
every artifact reflect the fuller dataset.

Usage
-----
    python -m scripts.augment_dataset
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.src.modeling.config import (
    CITY_TO_STATE,
    DATA_DIR,
    RANDOM_STATE,
    TIMESTAMP_COLUMN,
    TIMESTAMP_FORMAT,
)

#: The real (un-augmented) source dataset.
REAL_DATASET_PATH = DATA_DIR / "bank_transactions_data_2.csv"

#: The augmented dataset this script writes (config.DATASET_PATH).
FULL_DATASET_PATH = DATA_DIR / "bank_transactions_data_2_full.csv"

#: Standard deviation of the log-normal multiplicative jitter applied to the
#: continuous columns of every resampled synthetic row.
JITTER_SIGMA = 0.15

#: Half-width, in days, of the uniform shift applied to the resampled
#: ``TransactionDate`` of each synthetic row.
DATE_SHIFT_DAYS = 3.0

#: Continuous columns that receive the log-normal jitter, with the number of
#: decimal places to round to (matching how the source stores them).
CONTINUOUS_COLUMNS = {
    "TransactionAmount": 2,
    "AccountBalance": 2,
    "TransactionDuration": 0,
}

#: One assigned city per missing state. Each is the state's largest (or a
#: prominent) city, deliberately chosen so it collides with neither the real
#: 43 cities nor any other assigned city (asserted at runtime).
MISSING_STATE_CITY = {
    "Alabama": "Birmingham",
    "Alaska": "Anchorage",
    "Arkansas": "Little Rock",
    "Connecticut": "Bridgeport",
    "Delaware": "Wilmington",
    "Hawaii": "Honolulu",
    "Idaho": "Boise",
    "Iowa": "Des Moines",
    "Kansas": "Wichita",
    "Louisiana": "New Orleans",
    "Maine": "Bangor",
    "Minnesota": "Minneapolis",
    "Mississippi": "Jackson",
    "Montana": "Billings",
    "New Hampshire": "Manchester",
    "New Jersey": "Newark",
    "North Dakota": "Fargo",
    "Puerto Rico": "San Juan",
    "Rhode Island": "Providence",
    "South Carolina": "Columbia",
    "South Dakota": "Sioux Falls",
    "Utah": "Salt Lake City",
    "Vermont": "Burlington",
    "West Virginia": "Charleston",
    "Wyoming": "Cheyenne",
}

#: Approximate 2020 census populations, in millions, for the missing states.
#: Only used to size each state's synthetic sample; not a modelled quantity.
STATE_POPULATION_MILLIONS = {
    "Alabama": 5.02,
    "Alaska": 0.73,
    "Arkansas": 3.01,
    "Connecticut": 3.61,
    "Delaware": 0.99,
    "Hawaii": 1.46,
    "Idaho": 1.84,
    "Iowa": 3.19,
    "Kansas": 2.94,
    "Louisiana": 4.66,
    "Maine": 1.36,
    "Minnesota": 5.71,
    "Mississippi": 2.96,
    "Montana": 1.08,
    "New Hampshire": 1.38,
    "New Jersey": 9.29,
    "North Dakota": 0.78,
    "Puerto Rico": 3.28,
    "Rhode Island": 1.10,
    "South Carolina": 5.12,
    "South Dakota": 0.89,
    "Utah": 3.27,
    "Vermont": 0.64,
    "West Virginia": 1.79,
    "Wyoming": 0.58,
}

#: Bounds for the per-state synthetic row count.
MIN_ROWS_PER_STATE = 20
MAX_ROWS_PER_STATE = 100

#: Synthetic id formats.
SYNTHETIC_TXN_ID = "TXS{:06d}"
SYNTHETIC_ACCOUNT_ID = "ACS{:04d}"


def state_row_count(state: str) -> int:
    """
    Number of synthetic rows to generate for a state.

    Parameters
    ----------
    state : str
        State name, must be a key of :data:`STATE_POPULATION_MILLIONS`.

    Returns
    -------
    int
        ``clip(round(pop_millions * 10), MIN_ROWS_PER_STATE, MAX_ROWS_PER_STATE)``.
    """
    raw = round(STATE_POPULATION_MILLIONS[state] * 10)
    return int(np.clip(raw, MIN_ROWS_PER_STATE, MAX_ROWS_PER_STATE))


def _assert_no_city_collisions(real_cities: set[str]) -> None:
    """
    Fail loudly if an assigned city clashes with a real city or another
    assigned city.

    Parameters
    ----------
    real_cities : set of str
        The distinct ``Location`` values in the real dataset.

    Raises
    ------
    AssertionError
        On any collision, or if the missing-state set does not match the
        GeoJSON-derived expectation.
    """
    assigned = list(MISSING_STATE_CITY.values())

    duplicates = sorted({c for c in assigned if assigned.count(c) > 1})
    assert not duplicates, f"assigned cities collide with each other: {duplicates}"

    overlap = sorted(real_cities & set(assigned))
    assert not overlap, f"assigned cities collide with real cities: {overlap}"

    assert set(MISSING_STATE_CITY) == set(STATE_POPULATION_MILLIONS), (
        "MISSING_STATE_CITY and STATE_POPULATION_MILLIONS must cover the same "
        "states"
    )

    unmapped = {
        city: state
        for state, city in MISSING_STATE_CITY.items()
        if CITY_TO_STATE.get(city) != state
    }
    assert not unmapped, (
        f"config.CITY_TO_STATE is missing / disagrees on assigned cities: {unmapped}"
    )


def _jitter_continuous(
    values: np.ndarray,
    rng: np.random.Generator,
    low: float,
    high: float,
    decimals: int,
) -> np.ndarray:
    """
    Apply a log-normal multiplicative jitter, clip to observed bounds, round.

    Parameters
    ----------
    values : numpy.ndarray
        Resampled real values for one continuous column.
    rng : numpy.random.Generator
        Seeded generator.
    low, high : float
        Observed global minimum / maximum for the column in the real data.
    decimals : int
        Decimal places to round to (0 rounds to an integer value).

    Returns
    -------
    numpy.ndarray
        Jittered, clipped and rounded values.
    """
    factor = np.exp(rng.normal(0.0, JITTER_SIGMA, size=len(values)))
    out = np.clip(values.astype(float) * factor, low, high)
    out = np.round(out, decimals)
    if decimals == 0:
        out = out.astype("int64")
    return out


def build_synthetic_rows(real: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Generate the synthetic transactions for every missing state.

    Parameters
    ----------
    real : pandas.DataFrame
        The real dataset, exactly as read from :data:`REAL_DATASET_PATH`.

    Returns
    -------
    synthetic : pandas.DataFrame
        Synthetic rows, same columns / dtypes as ``real``.
    per_state_counts : dict of {str: int}
        Row count generated per state.
    """
    rng = np.random.default_rng(RANDOM_STATE)

    timestamps = pd.to_datetime(real[TIMESTAMP_COLUMN], format=TIMESTAMP_FORMAT)
    span_start, span_end = timestamps.min(), timestamps.max()

    bounds = {
        col: (float(real[col].min()), float(real[col].max()))
        for col in CONTINUOUS_COLUMNS
    }

    txn_counter = 0
    account_counter = 0
    frames: list[pd.DataFrame] = []
    per_state_counts: dict[str, int] = {}

    for state in sorted(MISSING_STATE_CITY):
        city = MISSING_STATE_CITY[state]
        n = state_row_count(state)
        per_state_counts[state] = n

        sample = real.sample(n=n, replace=True, random_state=rng.integers(0, 2**32))
        sample = sample.reset_index(drop=True)

        sample["Location"] = city

        sample["TransactionID"] = [
            SYNTHETIC_TXN_ID.format(txn_counter + i + 1) for i in range(n)
        ]
        txn_counter += n

        pool_size = math.ceil(n / 5)
        pool = [
            SYNTHETIC_ACCOUNT_ID.format(account_counter + i + 1)
            for i in range(pool_size)
        ]
        account_counter += pool_size
        sample["AccountID"] = rng.choice(pool, size=n)

        for col, decimals in CONTINUOUS_COLUMNS.items():
            low, high = bounds[col]
            sample[col] = _jitter_continuous(
                sample[col].to_numpy(), rng, low, high, decimals
            )

        base = pd.to_datetime(sample[TIMESTAMP_COLUMN], format=TIMESTAMP_FORMAT)
        shift = pd.to_timedelta(
            rng.uniform(-DATE_SHIFT_DAYS, DATE_SHIFT_DAYS, size=n), unit="D"
        )
        shifted = (base + shift).clip(lower=span_start, upper=span_end)
        sample[TIMESTAMP_COLUMN] = shifted.dt.strftime(TIMESTAMP_FORMAT)

        frames.append(sample[real.columns])

    synthetic = pd.concat(frames, ignore_index=True)
    synthetic = synthetic.astype(real.dtypes.to_dict())
    return synthetic, per_state_counts


def main() -> None:
    """Build the augmented dataset and write it next to the real one."""
    real = pd.read_csv(REAL_DATASET_PATH)
    real_cities = set(real["Location"].unique())

    _assert_no_city_collisions(real_cities)

    synthetic, per_state_counts = build_synthetic_rows(real)

    full = pd.concat([real, synthetic], ignore_index=True)
    full = full.astype(real.dtypes.to_dict())

    text = full.to_csv(index=False, lineterminator="\n")
    tmp_path = FULL_DATASET_PATH.with_suffix(FULL_DATASET_PATH.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(FULL_DATASET_PATH)

    # Guard: the real rows must survive concatenation byte-for-byte.
    real_roundtrip = real.to_csv(index=False, lineterminator="\n")
    full_head = "\n".join(text.splitlines()[: len(real) + 1]) + "\n"
    assert full_head == real_roundtrip, "real rows changed during augmentation"

    print(f"Real rows      : {len(real)}")
    print(f"Synthetic rows : {len(synthetic)}")
    print(f"New total      : {len(full)}")
    print()
    print("Per-state synthetic row counts:")
    for state in sorted(per_state_counts):
        print(
            f"  {state:<16s} {MISSING_STATE_CITY[state]:<16s} "
            f"{per_state_counts[state]:>4d}"
        )
    print()
    print(f"Wrote {FULL_DATASET_PATH}")


if __name__ == "__main__":
    main()
