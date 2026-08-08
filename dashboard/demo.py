"""API-backed helpers for the RiskFlow PayGuard product demonstration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import httpx
from pydantic import ValidationError

from api.schemas import (
    BatchPredictRequest,
    PredictionResponse,
    TransactionRequest,
)


DEFAULT_DEMO_BATCH_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "sample_payloads"
    / "predict_batch.json"
)


_SCENARIO_METADATA: dict[
    str,
    tuple[str, str],
] = {
    "demo-everyday-001": (
        "Everyday purchase",
        (
            "A small synthetic card purchase with matching "
            "purchase signals and no identity payload."
        ),
    ),
    "demo-higher-value-002": (
        "Higher-value purchase",
        (
            "A larger synthetic credit-card purchase with "
            "matching email domains and no identity payload."
        ),
    ),
    "demo-mobile-identity-004": (
        "Mobile identity-rich purchase",
        (
            "A synthetic mobile purchase with populated "
            "identity attributes."
        ),
    ),
}


class DemoDataError(RuntimeError):
    """Raised when committed demonstration data is unavailable or invalid."""


class DemoAPIError(RuntimeError):
    """Raised when the live FastAPI scoring path cannot return a valid result."""


@dataclass(frozen=True)
class DemoScenario:
    """One committed synthetic transaction exposed by the product demo."""

    label: str
    description: str
    transaction: TransactionRequest


def load_demo_scenarios(
    path: Path = DEFAULT_DEMO_BATCH_PATH,
) -> tuple[
    DemoScenario,
    ...,
]:
    """Load and validate the committed deterministic demonstration batch."""

    try:
        raw_text = path.read_text(
            encoding="utf-8"
        )
    except OSError as error:
        raise DemoDataError(
            "Committed demo payloads are unavailable: "
            f"{path}"
        ) from error

    try:
        raw_payload = json.loads(
            raw_text
        )
    except json.JSONDecodeError as error:
        raise DemoDataError(
            "Committed demo payload JSON is invalid"
        ) from error

    try:
        batch = (
            BatchPredictRequest
            .model_validate(
                raw_payload
            )
        )
    except ValidationError as error:
        raise DemoDataError(
            "Committed demo payloads do not match "
            "the frozen API request contract"
        ) from error

    expected_ids = tuple(
        _SCENARIO_METADATA
    )
    actual_ids = tuple(
        str(
            transaction.transaction_id
        )
        for transaction
        in batch.transactions
    )

    if actual_ids != expected_ids:
        raise DemoDataError(
            "Committed demo scenario IDs or order changed unexpectedly"
        )

    scenarios: list[
        DemoScenario
    ] = []

    for transaction in (
        batch.transactions
    ):
        transaction_id = str(
            transaction.transaction_id
        )

        (
            label,
            description,
        ) = _SCENARIO_METADATA[
            transaction_id
        ]

        scenarios.append(
            DemoScenario(
                label=label,
                description=description,
                transaction=transaction,
            )
        )

    return tuple(
        scenarios
    )


def submit_demo_prediction(
    transaction: TransactionRequest,
    api_base_url: str,
    *,
    timeout_seconds: float = 10.0,
) -> PredictionResponse:
    """Submit one demo transaction through the real FastAPI prediction path."""

    normalized_base_url = (
        api_base_url
        .strip()
        .rstrip("/")
    )

    if not normalized_base_url:
        raise DemoAPIError(
            "PayGuard API URL is empty"
        )

    prediction_url = (
        f"{normalized_base_url}/predict"
    )

    try:
        response = httpx.post(
            prediction_url,
            json=transaction.model_dump(
                mode="json"
            ),
            timeout=timeout_seconds,
        )
    except httpx.RequestError as error:
        raise DemoAPIError(
            "PayGuard API is unavailable at "
            f"{normalized_base_url}. "
            "Start the FastAPI service and try again. "
            "No local scoring fallback was used."
        ) from error

    if response.status_code != 200:
        raise DemoAPIError(
            "PayGuard API returned HTTP "
            f"{response.status_code} for /predict. "
            "No local scoring fallback was used."
        )

    try:
        response_payload = (
            response.json()
        )
    except ValueError as error:
        raise DemoAPIError(
            "PayGuard API returned a non-JSON prediction response"
        ) from error

    try:
        prediction = (
            PredictionResponse
            .model_validate(
                response_payload
            )
        )
    except ValidationError as error:
        raise DemoAPIError(
            "PayGuard API returned a response that does not "
            "match the frozen prediction contract"
        ) from error

    if (
        prediction.transaction_id
        != transaction.transaction_id
    ):
        raise DemoAPIError(
            "PayGuard API returned a mismatched transaction identifier"
        )

    return prediction
