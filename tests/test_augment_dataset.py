"""
Tests for ``scripts/augment_dataset.py``: the synthetic state-coverage
augmentation must add every missing state, keep its synthetic rows marked and
in-range, and leave the real rows untouched.
"""

import math

import numpy as np
import pandas as pd
import pytest

from backend.src.modeling.config import CITY_TO_STATE, GEOJSON_PATH
from scripts.augment_dataset import (
    CONTINUOUS_COLUMNS,
    FULL_DATASET_PATH,
    MISSING_STATE_CITY,
    REAL_DATASET_PATH,
    STATE_POPULATION_MILLIONS,
    build_synthetic_rows,
    state_row_count,
)

pytestmark = pytest.mark.skipif(
    not REAL_DATASET_PATH.exists(),
    reason=f"real dataset not found at {REAL_DATASET_PATH}",
)


@pytest.fixture(scope="module")
def real() -> pd.DataFrame:
    """The real, un-augmented dataset."""
    return pd.read_csv(REAL_DATASET_PATH)


@pytest.fixture(scope="module")
def synthetic(real: pd.DataFrame) -> pd.DataFrame:
    """Freshly generated synthetic rows (deterministic, random_state=42)."""
    frame, _ = build_synthetic_rows(real)
    return frame


def test_all_missing_states_get_a_city(synthetic: pd.DataFrame) -> None:
    """Every one of the 25 missing states appears in the synthetic rows."""
    import json

    geo = json.loads(GEOJSON_PATH.read_text(encoding="utf-8"))
    geo_names = {f["properties"]["name"] for f in geo["features"]}
    real_states = set(CITY_TO_STATE[c] for c in pd.read_csv(REAL_DATASET_PATH)["Location"].unique())
    missing = geo_names - real_states

    assert set(MISSING_STATE_CITY) == missing

    synth_states = {CITY_TO_STATE[c] for c in synthetic["Location"].unique()}
    assert missing <= synth_states


def test_augmented_dataset_covers_every_geojson_state(synthetic: pd.DataFrame) -> None:
    """Real + synthetic states span every GeoJSON polygon."""
    import json

    geo = json.loads(GEOJSON_PATH.read_text(encoding="utf-8"))
    geo_names = {f["properties"]["name"] for f in geo["features"]}

    real_locs = pd.read_csv(REAL_DATASET_PATH)["Location"].unique()
    covered = {CITY_TO_STATE[c] for c in real_locs}
    covered |= {CITY_TO_STATE[c] for c in synthetic["Location"].unique()}

    assert geo_names <= covered


def test_synthetic_ids_are_prefixed(synthetic: pd.DataFrame) -> None:
    """Synthetic transaction / account ids carry the TXS / ACS prefix."""
    assert synthetic["TransactionID"].str.startswith("TXS").all()
    assert synthetic["AccountID"].str.startswith("ACS").all()
    assert synthetic["TransactionID"].is_unique


def test_synthetic_account_pool_size(real: pd.DataFrame) -> None:
    """Each state's account pool is ceil(n / 5)."""
    frame, counts = build_synthetic_rows(real)
    frame = frame.copy()
    frame["state"] = frame["Location"].map(CITY_TO_STATE)
    for state, n in counts.items():
        pool = frame.loc[frame["state"] == state, "AccountID"].nunique()
        assert pool <= math.ceil(n / 5)


def test_continuous_columns_within_observed_range(
    real: pd.DataFrame, synthetic: pd.DataFrame
) -> None:
    """Jittered continuous columns stay inside the real observed [min, max]."""
    for col in CONTINUOUS_COLUMNS:
        lo, hi = float(real[col].min()), float(real[col].max())
        assert synthetic[col].min() >= lo - 1e-9
        assert synthetic[col].max() <= hi + 1e-9


def test_synthetic_dates_within_real_span(
    real: pd.DataFrame, synthetic: pd.DataFrame
) -> None:
    """Shifted timestamps never leave the real dataset's time span."""
    real_ts = pd.to_datetime(real["TransactionDate"])
    synth_ts = pd.to_datetime(synthetic["TransactionDate"])
    assert synth_ts.min() >= real_ts.min()
    assert synth_ts.max() <= real_ts.max()


def test_row_counts_follow_population_clip() -> None:
    """n = clip(round(pop_millions * 10), 20, 100)."""
    for state, pop in STATE_POPULATION_MILLIONS.items():
        assert state_row_count(state) == int(np.clip(round(pop * 10), 20, 100))


def test_schema_is_byte_identical(real: pd.DataFrame, synthetic: pd.DataFrame) -> None:
    """Synthetic rows carry exactly the source columns, in order, no extra."""
    assert list(synthetic.columns) == list(real.columns)


def test_real_rows_unchanged_on_disk(real: pd.DataFrame) -> None:
    """
    The committed augmented CSV keeps every real row byte-for-byte and adds
    only TXS-prefixed rows.
    """
    if not FULL_DATASET_PATH.exists():
        pytest.skip("augmented dataset not generated; run scripts.augment_dataset")

    full = pd.read_csv(FULL_DATASET_PATH)

    real_part = full[full["TransactionID"].str.startswith("TX") & ~full["TransactionID"].str.startswith("TXS")]
    assert len(real_part) == len(real)

    real_roundtrip = real.to_csv(index=False, lineterminator="\n")
    head = "\n".join(
        FULL_DATASET_PATH.read_text(encoding="utf-8").splitlines()[: len(real) + 1]
    ) + "\n"
    assert head == real_roundtrip

    synth_part = full[full["TransactionID"].str.startswith("TXS")]
    assert len(full) == len(real) + len(synth_part)


def test_city_to_state_covers_every_assigned_city() -> None:
    """config.CITY_TO_STATE maps every assigned placeholder city to its state."""
    for state, city in MISSING_STATE_CITY.items():
        assert CITY_TO_STATE.get(city) == state


def test_assigned_cities_do_not_collide() -> None:
    """Placeholder cities collide with neither the real 43 nor each other."""
    real_cities = set(pd.read_csv(REAL_DATASET_PATH)["Location"].unique())
    assigned = list(MISSING_STATE_CITY.values())

    assert len(assigned) == len(set(assigned))
    assert not (real_cities & set(assigned))
