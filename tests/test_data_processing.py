"""Tests for the IEEE-CIS data-loading contract."""

from pathlib import Path

import pandas as pd
import pytest

from src.data_processing import (
    AMOUNT_COLUMN,
    JOIN_KEY,
    TARGET_COLUMN,
    TIME_COLUMN,
    join_tables,
    load_test_tables,
    load_train_tables,
    normalize_identity_columns,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a small synthetic CSV fixture."""
    pd.DataFrame(rows).to_csv(path, index=False)


def test_normalize_identity_columns_uses_train_schema() -> None:
    identity = pd.DataFrame(
        {
            JOIN_KEY: [1],
            "id-01": [10.0],
            "id-02": [20.0],
            "DeviceType": ["desktop"],
        }
    )

    normalized = normalize_identity_columns(identity)

    assert "id_01" in normalized.columns
    assert "id_02" in normalized.columns
    assert "id-01" not in normalized.columns
    assert "id-02" not in normalized.columns
    assert "DeviceType" in normalized.columns


def test_join_tables_preserves_all_transaction_rows() -> None:
    transaction = pd.DataFrame(
        {
            JOIN_KEY: [1, 2, 3],
            TARGET_COLUMN: [0, 1, 0],
            TIME_COLUMN: [100, 200, 300],
            AMOUNT_COLUMN: [25.0, 50.0, 75.0],
        }
    )
    identity = pd.DataFrame(
        {
            JOIN_KEY: [2],
            "DeviceType": ["mobile"],
        }
    )

    joined = join_tables(transaction, identity)

    assert len(joined) == len(transaction)
    assert joined[JOIN_KEY].tolist() == [1, 2, 3]
    assert joined.loc[joined[JOIN_KEY] == 2, "DeviceType"].item() == "mobile"
    assert joined["DeviceType"].isna().sum() == 2


def test_join_tables_rejects_duplicate_identity_keys() -> None:
    transaction = pd.DataFrame({JOIN_KEY: [1, 2]})
    identity = pd.DataFrame(
        {
            JOIN_KEY: [1, 1],
            "DeviceType": ["mobile", "desktop"],
        }
    )

    with pytest.raises(ValueError, match="duplicate TransactionID"):
        join_tables(transaction, identity)


def test_load_train_tables_validates_contract(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "train_transaction.csv",
        [
            {
                JOIN_KEY: 1,
                TARGET_COLUMN: 0,
                TIME_COLUMN: 100,
                AMOUNT_COLUMN: 25.0,
                "ProductCD": "W",
            },
            {
                JOIN_KEY: 2,
                TARGET_COLUMN: 1,
                TIME_COLUMN: 200,
                AMOUNT_COLUMN: 80.0,
                "ProductCD": "C",
            },
        ],
    )
    _write_csv(
        tmp_path / "train_identity.csv",
        [
            {
                JOIN_KEY: 2,
                "id_01": 5.0,
                "DeviceType": "mobile",
            }
        ],
    )

    transaction, identity = load_train_tables(tmp_path)

    assert transaction.shape == (2, 5)
    assert identity.shape == (1, 3)
    assert transaction[JOIN_KEY].is_unique
    assert identity[JOIN_KEY].is_unique
    assert TARGET_COLUMN in transaction.columns


def test_load_test_tables_normalizes_identity_columns(
    tmp_path: Path,
) -> None:
    _write_csv(
        tmp_path / "test_transaction.csv",
        [
            {
                JOIN_KEY: 10,
                TIME_COLUMN: 500,
                AMOUNT_COLUMN: 120.0,
                "ProductCD": "R",
            }
        ],
    )
    _write_csv(
        tmp_path / "test_identity.csv",
        [
            {
                JOIN_KEY: 10,
                "id-01": 7.0,
                "id-02": 15.0,
            }
        ],
    )

    transaction, identity = load_test_tables(tmp_path)

    assert TARGET_COLUMN not in transaction.columns
    assert "id_01" in identity.columns
    assert "id_02" in identity.columns
    assert "id-01" not in identity.columns


def test_load_train_tables_rejects_missing_target(
    tmp_path: Path,
) -> None:
    _write_csv(
        tmp_path / "train_transaction.csv",
        [
            {
                JOIN_KEY: 1,
                TIME_COLUMN: 100,
                AMOUNT_COLUMN: 25.0,
            }
        ],
    )
    _write_csv(
        tmp_path / "train_identity.csv",
        [{JOIN_KEY: 1}],
    )

    with pytest.raises(ValueError, match="isFraud"):
        load_train_tables(tmp_path)


def test_load_train_tables_rejects_missing_source_file(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="train_transaction.csv"):
        load_train_tables(tmp_path)
