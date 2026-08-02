"""Build the compact Phase 1 fraud-model feature set.

The feature contract is based on the completed IEEE-CIS exploratory analysis.

The builder:

- selects the agreed transaction and identity fields
- preserves numerical missing values for tree-based models
- normalizes categorical missing values
- creates chronological and monetary derived features
- excludes keys, the target, raw time, anonymous V features, and deferred fields
- produces columns in a deterministic order

The transaction table remains the primary table. Identity fields are aligned
through a validated left join on TransactionID.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.data_processing import (
    AMOUNT_COLUMN,
    JOIN_KEY,
    TIME_COLUMN,
)


MISSING_CATEGORY = "<MISSING>"

SECONDS_PER_HOUR = 3_600
SECONDS_PER_DAY = 86_400


CORE_TRANSACTION_FEATURES: list[str] = [
    AMOUNT_COLUMN,
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "dist1",
    "P_emaildomain",
    "R_emaildomain",
]

COUNT_FEATURES: list[str] = [
    f"C{number}"
    for number in range(1, 15)
]

DELTA_FEATURES: list[str] = [
    "D1",
    "D2",
    "D3",
    "D4",
    "D10",
    "D11",
    "D15",
]

MATCH_FEATURES: list[str] = [
    "M1",
    "M2",
    "M3",
    "M4",
    "M6",
]

TRANSACTION_FEATURES: list[str] = [
    *CORE_TRANSACTION_FEATURES,
    *COUNT_FEATURES,
    *DELTA_FEATURES,
    *MATCH_FEATURES,
]

IDENTITY_FEATURES: list[str] = [
    "DeviceType",
    "id_01",
    "id_02",
    "id_05",
    "id_06",
    "id_11",
    "id_12",
    "id_13",
    "id_15",
    "id_16",
    "id_17",
    "id_19",
    "id_20",
    "id_28",
    "id_29",
    "id_31",
    "id_35",
    "id_36",
    "id_37",
    "id_38",
]

DERIVED_FEATURES: list[str] = [
    "transaction_amount_log1p",
    "relative_day",
    "relative_hour_cycle",
    "identity_availability",
]

FEATURE_COLUMNS: list[str] = [
    *TRANSACTION_FEATURES,
    *IDENTITY_FEATURES,
    *DERIVED_FEATURES,
]

CATEGORICAL_FEATURES: list[str] = [
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",
    *MATCH_FEATURES,
    "DeviceType",
    "id_12",
    "id_15",
    "id_16",
    "id_28",
    "id_29",
    "id_31",
    "id_35",
    "id_36",
    "id_37",
    "id_38",
    "relative_hour_cycle",
    "identity_availability",
]

NUMERICAL_FEATURES: list[str] = [
    column
    for column in FEATURE_COLUMNS
    if column not in CATEGORICAL_FEATURES
]


def _validate_required_columns(
    dataframe: pd.DataFrame,
    required_columns: Iterable[str],
    table_name: str,
) -> None:
    """Raise a clear error when required feature inputs are absent."""
    missing_columns = sorted(
        set(required_columns) - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{table_name} is missing required columns: "
            + ", ".join(missing_columns)
        )


def _validate_unique_key(
    dataframe: pd.DataFrame,
    table_name: str,
) -> None:
    """Require one row per TransactionID."""
    duplicate_count = int(
        dataframe[JOIN_KEY].duplicated().sum()
    )

    if duplicate_count:
        raise ValueError(
            f"{table_name} contains "
            f"{duplicate_count:,} duplicate {JOIN_KEY} values"
        )


def _validate_numeric_source(
    series: pd.Series,
    column_name: str,
    *,
    allow_negative: bool,
) -> pd.Series:
    """Convert a required numerical source and validate finite values."""
    try:
        numeric = pd.to_numeric(
            series,
            errors="raise",
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{column_name} must contain numerical values"
        ) from error

    if numeric.isna().any():
        raise ValueError(
            f"{column_name} must not contain missing values"
        )

    numeric_values = numeric.to_numpy(
        dtype="float64",
        na_value=np.nan,
    )

    if not np.isfinite(numeric_values).all():
        raise ValueError(
            f"{column_name} must contain only finite values"
        )

    if not allow_negative and (numeric < 0).any():
        raise ValueError(
            f"{column_name} must not contain negative values"
        )

    return numeric


def _resolve_time_origin(
    transaction_time: pd.Series,
    time_origin: float | int | None,
) -> float:
    """Resolve and validate the origin used for relative time features."""
    if time_origin is None:
        resolved_origin = float(
            transaction_time.min()
        )
    else:
        resolved_origin = float(time_origin)

    if not np.isfinite(resolved_origin):
        raise ValueError(
            "time_origin must be a finite numerical value"
        )

    if (transaction_time < resolved_origin).any():
        raise ValueError(
            "TransactionDT contains values earlier than time_origin"
        )

    return resolved_origin


def _align_identity_features(
    transaction: pd.DataFrame,
    identity: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Align optional identity fields to transaction row order."""
    if identity is None:
        aligned_identity = pd.DataFrame(
            {
                column: np.nan
                for column in IDENTITY_FEATURES
            },
            index=transaction.index,
        )

        identity_availability = pd.Series(
            0,
            index=transaction.index,
            dtype="int8",
        )

        return aligned_identity, identity_availability

    _validate_required_columns(
        identity,
        [JOIN_KEY, *IDENTITY_FEATURES],
        "identity",
    )
    _validate_unique_key(identity, "identity")

    identity_subset = identity[
        [JOIN_KEY, *IDENTITY_FEATURES]
    ].copy()

    aligned = transaction[[JOIN_KEY]].merge(
        identity_subset,
        on=JOIN_KEY,
        how="left",
        sort=False,
        validate="one_to_one",
        indicator="_identity_merge",
    )

    if len(aligned) != len(transaction):
        raise RuntimeError(
            "Transaction row count changed while aligning identity data"
        )

    identity_availability = (
        aligned["_identity_merge"]
        .eq("both")
        .astype("int8")
    )
    identity_availability.index = transaction.index

    aligned_identity = aligned[
        IDENTITY_FEATURES
    ].copy()
    aligned_identity.index = transaction.index

    return aligned_identity, identity_availability


def _normalize_categorical_features(
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Represent categorical values and missingness consistently."""
    normalized = features.copy()

    for column in CATEGORICAL_FEATURES:
        normalized[column] = (
            normalized[column]
            .astype("string")
            .fillna(MISSING_CATEGORY)
        )

    return normalized


def build_features(
    transaction: pd.DataFrame,
    identity: pd.DataFrame | None = None,
    *,
    time_origin: float | int | None = None,
) -> pd.DataFrame:
    """Build the deterministic 63-column Phase 1 feature matrix.

    Parameters
    ----------
    transaction:
        Primary transaction table containing one row per TransactionID.
    identity:
        Optional identity table containing zero or one row per TransactionID.
        When omitted, identity fields are represented as missing and
        ``identity_availability`` is zero.
    time_origin:
        Origin for relative time features. When omitted, the minimum
        TransactionDT in ``transaction`` is used. Production and split
        materialization code should pass one shared training-data origin.

    Returns
    -------
    pandas.DataFrame
        Feature matrix with columns exactly matching ``FEATURE_COLUMNS``.
    """
    if transaction.empty:
        raise ValueError(
            "transaction must contain at least one row"
        )

    _validate_required_columns(
        transaction,
        [
            JOIN_KEY,
            TIME_COLUMN,
            *TRANSACTION_FEATURES,
        ],
        "transaction",
    )
    _validate_unique_key(transaction, "transaction")

    transaction_amount = _validate_numeric_source(
        transaction[AMOUNT_COLUMN],
        AMOUNT_COLUMN,
        allow_negative=False,
    )

    transaction_time = _validate_numeric_source(
        transaction[TIME_COLUMN],
        TIME_COLUMN,
        allow_negative=False,
    )

    resolved_time_origin = _resolve_time_origin(
        transaction_time,
        time_origin,
    )

    features = transaction[
        TRANSACTION_FEATURES
    ].copy()

    (
        aligned_identity,
        identity_availability,
    ) = _align_identity_features(
        transaction,
        identity,
    )

    for column in IDENTITY_FEATURES:
        features[column] = aligned_identity[column]

    relative_seconds = (
        transaction_time.astype("float64")
        - resolved_time_origin
    )

    features["transaction_amount_log1p"] = np.log1p(
        transaction_amount.astype("float64")
    )

    features["relative_day"] = np.floor_divide(
        relative_seconds,
        SECONDS_PER_DAY,
    ).astype("int32")

    features["relative_hour_cycle"] = np.floor_divide(
        relative_seconds,
        SECONDS_PER_HOUR,
    ).mod(24).astype("int8")

    features["identity_availability"] = (
        identity_availability
    )

    features = _normalize_categorical_features(
        features
    )

    features = features.loc[
        :,
        FEATURE_COLUMNS,
    ]

    if len(features) != len(transaction):
        raise RuntimeError(
            "Feature row count does not match transaction row count"
        )

    if list(features.columns) != FEATURE_COLUMNS:
        raise RuntimeError(
            "Feature output does not match the declared feature contract"
        )

    features.attrs["time_origin"] = (
        resolved_time_origin
    )

    return features


def build_feature_metadata() -> pd.DataFrame:
    """Return one metadata row for every baseline feature."""
    group_lookup: dict[str, str] = {}

    for column in CORE_TRANSACTION_FEATURES:
        group_lookup[column] = "core_transaction"

    for column in COUNT_FEATURES:
        group_lookup[column] = "count_C"

    for column in DELTA_FEATURES:
        group_lookup[column] = "delta_D"

    for column in MATCH_FEATURES:
        group_lookup[column] = "match_M"

    for column in IDENTITY_FEATURES:
        group_lookup[column] = "identity"

    for column in DERIVED_FEATURES:
        group_lookup[column] = "derived"

    records = []

    for position, column in enumerate(
        FEATURE_COLUMNS,
        start=1,
    ):
        records.append(
            {
                "feature_position": position,
                "feature_name": column,
                "feature_group": group_lookup[column],
                "feature_type": (
                    "categorical"
                    if column in CATEGORICAL_FEATURES
                    else "numerical"
                ),
                "source": (
                    "derived"
                    if column in DERIVED_FEATURES
                    else (
                        "identity"
                        if column in IDENTITY_FEATURES
                        else "transaction"
                    )
                ),
            }
        )

    metadata = pd.DataFrame.from_records(records)

    if len(metadata) != len(FEATURE_COLUMNS):
        raise RuntimeError(
            "Feature metadata does not match the feature contract"
        )

    return metadata
