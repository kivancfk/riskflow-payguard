#!/usr/bin/env python3
"""Materialize the validated Phase 1 fraud-model datasets.

Outputs are written under ``data/processed`` and remain excluded from Git:

- train.parquet
- validation.parquet
- test.parquet
- kaggle_test.parquet
- feature_metadata.csv
- dataset_manifest.json

The internal labeled datasets use a stable chronological 70%/15%/15% split.
The Kaggle test set is processed separately and remains unlabeled.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import pandas as pd
import pyarrow.parquet as pq

from src.data_processing import (
    AMOUNT_COLUMN,
    JOIN_KEY,
    TARGET_COLUMN,
    TIME_COLUMN,
    load_test_tables,
    load_train_tables,
)
from src.features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    build_feature_metadata,
    build_features,
)


TRAIN_OUTPUT = "train.parquet"
VALIDATION_OUTPUT = "validation.parquet"
TEST_OUTPUT = "test.parquet"
KAGGLE_TEST_OUTPUT = "kaggle_test.parquet"
METADATA_OUTPUT = "feature_metadata.csv"
MANIFEST_OUTPUT = "dataset_manifest.json"

OUTPUT_FILENAMES = [
    TRAIN_OUTPUT,
    VALIDATION_OUTPUT,
    TEST_OUTPUT,
    KAGGLE_TEST_OUTPUT,
    METADATA_OUTPUT,
    MANIFEST_OUTPUT,
]

SPLIT_RATIOS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15,
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Build the Phase 1 IEEE-CIS baseline datasets."
        )
    )

    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing the raw IEEE-CIS CSV files.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for generated Phase 1 artifacts.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing generated artifacts.",
    )

    return parser.parse_args()


def _validate_output_policy(
    processed_dir: Path,
    *,
    overwrite: bool,
) -> None:
    """Prevent accidental replacement unless explicitly requested."""
    existing_outputs = [
        processed_dir / filename
        for filename in OUTPUT_FILENAMES
        if (processed_dir / filename).exists()
    ]

    if existing_outputs and not overwrite:
        existing_names = ", ".join(
            path.name for path in existing_outputs
        )

        raise FileExistsError(
            "Processed artifacts already exist: "
            f"{existing_names}. Use --overwrite to replace them."
        )


def _chronological_split_indices(
    transaction: pd.DataFrame,
) -> dict[str, pd.Index]:
    """Return stable chronological row indices for 70/15/15 splits."""
    sorted_index = (
        transaction[TIME_COLUMN]
        .sort_values(kind="stable")
        .index
    )

    row_count = len(transaction)
    train_end = int(row_count * SPLIT_RATIOS["train"])
    validation_end = int(
        row_count
        * (
            SPLIT_RATIOS["train"]
            + SPLIT_RATIOS["validation"]
        )
    )

    split_indices = {
        "train": sorted_index[:train_end],
        "validation": sorted_index[
            train_end:validation_end
        ],
        "test": sorted_index[validation_end:],
    }

    assigned_rows = sum(
        len(indices)
        for indices in split_indices.values()
    )

    if assigned_rows != row_count:
        raise RuntimeError(
            "Chronological split row count does not match source data"
        )

    return split_indices


def _build_labeled_output(
    transaction: pd.DataFrame,
    features: pd.DataFrame,
    indices: pd.Index,
) -> pd.DataFrame:
    """Build one labeled output split in chronological order."""
    output = features.loc[indices].copy()

    output.insert(
        0,
        TARGET_COLUMN,
        transaction.loc[indices, TARGET_COLUMN].to_numpy(),
    )
    output.insert(
        0,
        JOIN_KEY,
        transaction.loc[indices, JOIN_KEY].to_numpy(),
    )

    output = output.reset_index(drop=True)

    expected_columns = [
        JOIN_KEY,
        TARGET_COLUMN,
        *FEATURE_COLUMNS,
    ]

    if list(output.columns) != expected_columns:
        raise RuntimeError(
            "Labeled output does not match the expected schema"
        )

    return output


def _build_unlabeled_output(
    transaction: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Build the Kaggle scoring dataset."""
    output = features.copy()

    output.insert(
        0,
        JOIN_KEY,
        transaction[JOIN_KEY].to_numpy(),
    )

    output = output.reset_index(drop=True)

    expected_columns = [
        JOIN_KEY,
        *FEATURE_COLUMNS,
    ]

    if list(output.columns) != expected_columns:
        raise RuntimeError(
            "Kaggle output does not match the expected schema"
        )

    return output


def _write_parquet(
    dataframe: pd.DataFrame,
    path: Path,
) -> None:
    """Write one compressed Parquet artifact."""
    dataframe.to_parquet(
        path,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )


def _parquet_artifact_summary(
    path: Path,
) -> dict[str, Any]:
    """Return structural metadata without loading the full artifact."""
    parquet_file = pq.ParquetFile(path)
    metadata = parquet_file.metadata

    return {
        "filename": path.name,
        "row_count": int(metadata.num_rows),
        "column_count": int(metadata.num_columns),
        "size_bytes": int(path.stat().st_size),
    }


def _validate_parquet_schema(
    path: Path,
    expected_columns: list[str],
    expected_rows: int,
) -> None:
    """Validate an output Parquet artifact using file metadata."""
    actual_columns = pq.read_schema(path).names
    actual_rows = int(
        pq.ParquetFile(path).metadata.num_rows
    )

    if actual_columns != expected_columns:
        raise RuntimeError(
            f"{path.name} has an unexpected column schema"
        )

    if actual_rows != expected_rows:
        raise RuntimeError(
            f"{path.name} has {actual_rows:,} rows; "
            f"expected {expected_rows:,}"
        )


def _split_summary(
    transaction: pd.DataFrame,
    features: pd.DataFrame,
    indices: pd.Index,
) -> dict[str, Any]:
    """Describe one internal chronological split."""
    split_transaction = transaction.loc[indices]
    split_features = features.loc[indices]

    fraud_count = int(
        split_transaction[TARGET_COLUMN].sum()
    )
    transaction_count = len(split_transaction)

    identity_count = int(
        (
            split_features["identity_availability"]
            == "1"
        ).sum()
    )

    return {
        "transaction_count": transaction_count,
        "fraud_count": fraud_count,
        "fraud_rate": (
            fraud_count / transaction_count
            if transaction_count
            else 0.0
        ),
        "identity_count": identity_count,
        "identity_coverage": (
            identity_count / transaction_count
            if transaction_count
            else 0.0
        ),
        "minimum_transaction_dt": int(
            split_transaction[TIME_COLUMN].min()
        ),
        "maximum_transaction_dt": int(
            split_transaction[TIME_COLUMN].max()
        ),
        "total_transaction_amount": float(
            split_transaction[AMOUNT_COLUMN].sum()
        ),
        "fraudulent_transaction_amount": float(
            split_transaction.loc[
                split_transaction[TARGET_COLUMN] == 1,
                AMOUNT_COLUMN,
            ].sum()
        ),
    }


def _move_completed_artifacts(
    temporary_dir: Path,
    processed_dir: Path,
) -> None:
    """Move validated temporary outputs into their final locations."""
    for filename in OUTPUT_FILENAMES:
        source = temporary_dir / filename
        destination = processed_dir / filename

        if not source.exists():
            raise RuntimeError(
                f"Expected temporary artifact is missing: {source}"
            )

        source.replace(destination)


def materialize_phase1(
    raw_dir: Path,
    processed_dir: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build and validate all Phase 1 processed artifacts."""
    raw_dir = raw_dir.resolve()
    processed_dir = processed_dir.resolve()

    processed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    _validate_output_policy(
        processed_dir,
        overwrite=overwrite,
    )

    print("Loading labeled training sources...")
    transaction, identity = load_train_tables(raw_dir)

    time_origin = float(
        transaction[TIME_COLUMN].min()
    )

    split_indices = _chronological_split_indices(
        transaction
    )

    print("Building the 63-feature labeled matrix...")
    features = build_features(
        transaction,
        identity,
        time_origin=time_origin,
    )

    train_identity_coverage = (
        transaction[JOIN_KEY]
        .isin(identity[JOIN_KEY])
        .mean()
    )

    split_summaries = {
        split_name: _split_summary(
            transaction,
            features,
            indices,
        )
        for split_name, indices in split_indices.items()
    }

    source_summary: dict[str, Any] = {
        "train_transaction_rows": len(transaction),
        "train_identity_rows": len(identity),
        "train_identity_coverage": float(
            train_identity_coverage
        ),
    }

    labeled_expected_columns = [
        JOIN_KEY,
        TARGET_COLUMN,
        *FEATURE_COLUMNS,
    ]
    kaggle_expected_columns = [
        JOIN_KEY,
        *FEATURE_COLUMNS,
    ]

    with tempfile.TemporaryDirectory(
        prefix=".phase1-materialize-",
        dir=processed_dir,
    ) as temporary_name:
        temporary_dir = Path(temporary_name)

        artifact_summaries: dict[str, Any] = {}

        for split_name, output_name in [
            ("train", TRAIN_OUTPUT),
            ("validation", VALIDATION_OUTPUT),
            ("test", TEST_OUTPUT),
        ]:
            print(
                f"Writing internal {split_name} split..."
            )

            split_output = _build_labeled_output(
                transaction,
                features,
                split_indices[split_name],
            )

            output_path = temporary_dir / output_name

            _write_parquet(
                split_output,
                output_path,
            )

            _validate_parquet_schema(
                output_path,
                labeled_expected_columns,
                len(split_output),
            )

            artifact_summaries[split_name] = (
                _parquet_artifact_summary(output_path)
            )

            del split_output
            gc.collect()

        feature_metadata = build_feature_metadata()
        metadata_path = temporary_dir / METADATA_OUTPUT

        feature_metadata.to_csv(
            metadata_path,
            index=False,
        )

        if len(feature_metadata) != len(FEATURE_COLUMNS):
            raise RuntimeError(
                "Feature metadata row count is invalid"
            )

        artifact_summaries["feature_metadata"] = {
            "filename": METADATA_OUTPUT,
            "row_count": len(feature_metadata),
            "column_count": feature_metadata.shape[1],
            "size_bytes": int(
                metadata_path.stat().st_size
            ),
        }

        del features
        del transaction
        del identity
        gc.collect()

        print("Loading unlabeled Kaggle test sources...")
        test_transaction, test_identity = (
            load_test_tables(raw_dir)
        )

        test_identity_coverage = (
            test_transaction[JOIN_KEY]
            .isin(test_identity[JOIN_KEY])
            .mean()
        )

        print("Building the Kaggle scoring matrix...")
        test_features = build_features(
            test_transaction,
            test_identity,
            time_origin=time_origin,
        )

        kaggle_output = _build_unlabeled_output(
            test_transaction,
            test_features,
        )

        kaggle_output_path = (
            temporary_dir / KAGGLE_TEST_OUTPUT
        )

        _write_parquet(
            kaggle_output,
            kaggle_output_path,
        )

        _validate_parquet_schema(
            kaggle_output_path,
            kaggle_expected_columns,
            len(kaggle_output),
        )

        artifact_summaries["kaggle_test"] = (
            _parquet_artifact_summary(
                kaggle_output_path
            )
        )

        source_summary.update(
            {
                "kaggle_test_transaction_rows": len(
                    test_transaction
                ),
                "kaggle_test_identity_rows": len(
                    test_identity
                ),
                "kaggle_test_identity_coverage": float(
                    test_identity_coverage
                ),
            }
        )

        del kaggle_output
        del test_features
        del test_transaction
        del test_identity
        gc.collect()

        manifest = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "source_dataset": (
                "IEEE-CIS Fraud Detection"
            ),
            "time_origin_transaction_dt": time_origin,
            "split_contract": {
                "strategy": (
                    "stable chronological ordering by "
                    "TransactionDT"
                ),
                "ratios": SPLIT_RATIOS,
                "splits": split_summaries,
            },
            "feature_contract": {
                "feature_count": len(FEATURE_COLUMNS),
                "categorical_feature_count": len(
                    CATEGORICAL_FEATURES
                ),
                "numerical_feature_count": len(
                    NUMERICAL_FEATURES
                ),
                "feature_columns": FEATURE_COLUMNS,
                "categorical_features": (
                    CATEGORICAL_FEATURES
                ),
                "numerical_features": NUMERICAL_FEATURES,
                "labeled_output_columns": (
                    labeled_expected_columns
                ),
                "kaggle_output_columns": (
                    kaggle_expected_columns
                ),
            },
            "source_summary": source_summary,
            "artifacts": artifact_summaries,
        }

        manifest_path = temporary_dir / MANIFEST_OUTPUT

        manifest_path.write_text(
            json.dumps(
                manifest,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        parsed_manifest = json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )

        if parsed_manifest["feature_contract"][
            "feature_count"
        ] != len(FEATURE_COLUMNS):
            raise RuntimeError(
                "Generated manifest failed validation"
            )

        _move_completed_artifacts(
            temporary_dir,
            processed_dir,
        )

    print(
        f"Phase 1 artifacts created in "
        f"{processed_dir}"
    )

    return manifest


def main() -> None:
    """Run the Phase 1 materialization workflow."""
    args = parse_args()

    manifest = materialize_phase1(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        overwrite=args.overwrite,
    )

    print("\nMaterialization summary")

    for split_name in [
        "train",
        "validation",
        "test",
    ]:
        summary = manifest["split_contract"][
            "splits"
        ][split_name]

        print(
            f"{split_name:>10}: "
            f"{summary['transaction_count']:,} rows, "
            f"{summary['fraud_rate']:.4%} fraud"
        )

    kaggle_rows = manifest["source_summary"][
        "kaggle_test_transaction_rows"
    ]

    print(
        f"{'kaggle_test':>10}: "
        f"{kaggle_rows:,} rows"
    )


if __name__ == "__main__":
    main()
