"""Load and validate the IEEE-CIS Fraud Detection source tables.

The transaction table is the primary table. Identity data is optional and is
left-joined by TransactionID because only a subset of transactions has an
identity record.

The Kaggle test identity table uses names such as ``id-01``, while the train
identity table uses ``id_01``. Identity column names are normalized to the
underscore convention when loaded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


JOIN_KEY = "TransactionID"
TARGET_COLUMN = "isFraud"
TIME_COLUMN = "TransactionDT"
AMOUNT_COLUMN = "TransactionAmt"

TRAIN_TRANSACTION_FILE = "train_transaction.csv"
TRAIN_IDENTITY_FILE = "train_identity.csv"
TEST_TRANSACTION_FILE = "test_transaction.csv"
TEST_IDENTITY_FILE = "test_identity.csv"

PathLike = str | Path


def _validate_source_file(path: Path) -> None:
    """Raise a clear error when a required source file is unavailable."""
    if not path.exists():
        raise FileNotFoundError(f"Required IEEE-CIS file not found: {path}")

    if not path.is_file():
        raise FileNotFoundError(f"Expected a file but found another path: {path}")

    if path.stat().st_size == 0:
        raise ValueError(f"Required IEEE-CIS file is empty: {path}")


def _validate_columns(
    dataframe: pd.DataFrame,
    required_columns: Iterable[str],
    table_name: str,
) -> None:
    """Validate the minimum columns required by the data contract."""
    missing = sorted(set(required_columns) - set(dataframe.columns))

    if missing:
        raise ValueError(
            f"{table_name} is missing required columns: {', '.join(missing)}"
        )


def _validate_unique_key(
    dataframe: pd.DataFrame,
    table_name: str,
    key: str = JOIN_KEY,
) -> None:
    """Ensure the source table has one row per transaction key."""
    duplicate_count = int(dataframe[key].duplicated().sum())

    if duplicate_count:
        raise ValueError(
            f"{table_name} contains {duplicate_count:,} duplicate {key} values"
        )


def normalize_identity_columns(identity: pd.DataFrame) -> pd.DataFrame:
    """Normalize Kaggle test identity names from id-01 to id_01."""
    rename_map = {
        column: column.replace("id-", "id_", 1)
        for column in identity.columns
        if column.startswith("id-")
    }

    normalized = identity.rename(columns=rename_map).copy()

    duplicated_columns = normalized.columns[
        normalized.columns.duplicated()
    ].tolist()

    if duplicated_columns:
        raise ValueError(
            "Identity column normalization created duplicate columns: "
            + ", ".join(duplicated_columns)
        )

    return normalized


def _read_csv(
    path: Path,
    *,
    nrows: int | None = None,
    usecols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Read one validated source CSV."""
    _validate_source_file(path)

    return pd.read_csv(
        path,
        nrows=nrows,
        usecols=list(usecols) if usecols is not None else None,
    )


def load_train_tables(
    data_dir: PathLike = "data/raw",
    *,
    nrows: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate the IEEE-CIS train transaction and identity tables."""
    raw_dir = Path(data_dir)

    transaction = _read_csv(
        raw_dir / TRAIN_TRANSACTION_FILE,
        nrows=nrows,
    )
    identity = _read_csv(
        raw_dir / TRAIN_IDENTITY_FILE,
        nrows=nrows,
    )
    identity = normalize_identity_columns(identity)

    _validate_columns(
        transaction,
        required_columns=(
            JOIN_KEY,
            TARGET_COLUMN,
            TIME_COLUMN,
            AMOUNT_COLUMN,
        ),
        table_name=TRAIN_TRANSACTION_FILE,
    )
    _validate_columns(
        identity,
        required_columns=(JOIN_KEY,),
        table_name=TRAIN_IDENTITY_FILE,
    )

    _validate_unique_key(transaction, TRAIN_TRANSACTION_FILE)
    _validate_unique_key(identity, TRAIN_IDENTITY_FILE)

    return transaction, identity


def load_test_tables(
    data_dir: PathLike = "data/raw",
    *,
    nrows: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate the IEEE-CIS test transaction and identity tables."""
    raw_dir = Path(data_dir)

    transaction = _read_csv(
        raw_dir / TEST_TRANSACTION_FILE,
        nrows=nrows,
    )
    identity = _read_csv(
        raw_dir / TEST_IDENTITY_FILE,
        nrows=nrows,
    )
    identity = normalize_identity_columns(identity)

    _validate_columns(
        transaction,
        required_columns=(JOIN_KEY, TIME_COLUMN, AMOUNT_COLUMN),
        table_name=TEST_TRANSACTION_FILE,
    )
    _validate_columns(
        identity,
        required_columns=(JOIN_KEY,),
        table_name=TEST_IDENTITY_FILE,
    )

    _validate_unique_key(transaction, TEST_TRANSACTION_FILE)
    _validate_unique_key(identity, TEST_IDENTITY_FILE)

    if TARGET_COLUMN in transaction.columns:
        raise ValueError(
            f"{TEST_TRANSACTION_FILE} unexpectedly contains {TARGET_COLUMN}"
        )

    return transaction, identity


def join_tables(
    transaction: pd.DataFrame,
    identity: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join optional identity attributes onto the transaction table."""
    _validate_columns(
        transaction,
        required_columns=(JOIN_KEY,),
        table_name="transaction",
    )
    _validate_columns(
        identity,
        required_columns=(JOIN_KEY,),
        table_name="identity",
    )

    _validate_unique_key(transaction, "transaction")
    _validate_unique_key(identity, "identity")

    joined = transaction.merge(
        identity,
        on=JOIN_KEY,
        how="left",
        validate="one_to_one",
    )

    if len(joined) != len(transaction):
        raise RuntimeError(
            "Transaction row count changed after the identity left join"
        )

    return joined


def load_raw(
    data_dir: PathLike = "data/raw",
    *,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Load and join the train tables.

    This function preserves the original module interface. New code may use
    ``load_train_tables`` and ``join_tables`` separately when table-level
    analysis is required.
    """
    transaction, identity = load_train_tables(data_dir, nrows=nrows)
    return join_tables(transaction, identity)