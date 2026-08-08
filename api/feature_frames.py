"""Construct exact ordered model feature frames from validated API requests."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from api.schemas import (
    TransactionModelFeatures,
    TransactionRequest,
)
from src.features import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
)
from src.policy_bundle import (
    CalibratedPolicyBundle,
)


TransactionIdentifier = str | int


def _validate_api_feature_contract(
    bundle: CalibratedPolicyBundle,
) -> None:
    """Require the API schema to match the loaded frozen model contract."""

    if not isinstance(
        bundle,
        CalibratedPolicyBundle,
    ):
        raise TypeError(
            "bundle must be a CalibratedPolicyBundle"
        )

    baseline_bundle = (
        bundle.baseline_bundle
    )

    api_feature_names = tuple(
        TransactionModelFeatures
        .model_fields
        .keys()
    )

    if set(api_feature_names) != set(
        baseline_bundle.feature_columns
    ):
        missing_from_api = sorted(
            set(
                baseline_bundle.feature_columns
            )
            - set(api_feature_names)
        )
        extra_in_api = sorted(
            set(api_feature_names)
            - set(
                baseline_bundle.feature_columns
            )
        )

        raise RuntimeError(
            "API feature names do not match the frozen model contract. "
            f"missing_from_api={missing_from_api}, "
            f"extra_in_api={extra_in_api}"
        )

    if set(CATEGORICAL_FEATURES) != set(
        baseline_bundle.categorical_features
    ):
        raise RuntimeError(
            "API categorical features do not match "
            "the frozen model contract"
        )

    if set(NUMERICAL_FEATURES) != set(
        baseline_bundle.numerical_features
    ):
        raise RuntimeError(
            "API numerical features do not match "
            "the frozen model contract"
        )


def _validate_unique_transaction_ids(
    transactions: Sequence[
        TransactionRequest
    ],
) -> tuple[
    TransactionIdentifier,
    ...,
]:
    """Return aligned unique transaction identifiers."""

    transaction_ids = tuple(
        transaction.transaction_id
        for transaction in transactions
    )

    if len(transaction_ids) != len(
        set(transaction_ids)
    ):
        raise ValueError(
            "transaction_id values must be unique within a batch"
        )

    return transaction_ids


def build_batch_feature_frame(
    transactions: Sequence[
        TransactionRequest
    ],
    bundle: CalibratedPolicyBundle,
) -> tuple[
    tuple[
        TransactionIdentifier,
        ...,
    ],
    pd.DataFrame,
]:
    """Build one exact-order frozen-model frame for an ordered request batch."""

    if not transactions:
        raise ValueError(
            "transactions must contain at least one transaction"
        )

    _validate_api_feature_contract(
        bundle
    )

    transaction_ids = (
        _validate_unique_transaction_ids(
            transactions
        )
    )

    records = [
        transaction.features.model_dump(
            mode="python"
        )
        for transaction in transactions
    ]

    baseline_bundle = (
        bundle.baseline_bundle
    )
    ordered_columns = list(
        baseline_bundle.feature_columns
    )

    feature_frame = pd.DataFrame.from_records(
        records
    ).loc[
        :,
        ordered_columns,
    ].copy()

    for feature_name in (
        baseline_bundle.numerical_features
    ):
        feature_frame[
            feature_name
        ] = pd.to_numeric(
            feature_frame[
                feature_name
            ],
            errors="raise",
        ).astype(
            "float64"
        )

    return (
        transaction_ids,
        feature_frame,
    )


def build_feature_frame(
    transaction: TransactionRequest,
    bundle: CalibratedPolicyBundle,
) -> pd.DataFrame:
    """Build an exact-order one-row frozen-model feature frame."""

    _, feature_frame = (
        build_batch_feature_frame(
            [transaction],
            bundle,
        )
    )

    return feature_frame
