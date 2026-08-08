#!/usr/bin/env python3
"""End-to-end Docker Compose deployment smoke validation."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

PROJECT_NAME = "riskflow-payguard-smoke"

COMPOSE = [
    "docker",
    "compose",
    "--project-name",
    PROJECT_NAME,
]

API_BASE_URL = "http://localhost:8000"
DASHBOARD_HEALTH_URL = (
    "http://localhost:8501/_stcore/health"
)

FROZEN_BASELINE_MODEL_VERSION = "baseline-v1"
FROZEN_POLICY_VERSION = "calibrated-policy-v1"
FROZEN_CALIBRATION_METHOD = "sigmoid"

FROZEN_REVIEW_THRESHOLD = (
    0.16255069862369795
)
FROZEN_BLOCK_THRESHOLD = (
    0.8509223095305902
)

FROZEN_EXPLANATION_VERSION = (
    "shap-explanation-v1"
)
FROZEN_REASON_CODE_VERSION = (
    "reason-codes-v1"
)

FROZEN_POLICY_SHA256 = (
    "5d53f23719ae891ecc24585393585765aa"
    "7fc0900ab38f95e37f59c18fe6c90f"
)

STRING_TRANSACTION_ID = (
    "phase7-smoke-string"
)
INTEGER_TRANSACTION_ID = 7001


def run(
    command: list[str],
    *,
    capture_output: bool = False,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run one checked command from the repository root."""

    print(
        "$",
        shlex.join(command),
        flush=True,
    )

    return subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        text=True,
        input=input_text,
        capture_output=capture_output,
    )


def compose(
    *arguments: str,
    capture_output: bool = False,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run one Docker Compose command in the isolated smoke project."""

    return run(
        [
            *COMPOSE,
            *arguments,
        ],
        capture_output=capture_output,
        input_text=input_text,
    )


def get_json(
    path: str,
) -> dict[str, object]:
    """GET one API endpoint and decode its JSON response."""

    request = urllib.request.Request(
        f"{API_BASE_URL}{path}",
        method="GET",
    )

    with urllib.request.urlopen(
        request,
        timeout=30,
    ) as response:
        if response.status != 200:
            raise RuntimeError(
                f"GET {path} returned "
                f"HTTP {response.status}"
            )

        return json.loads(
            response.read()
        )


def post_json(
    path: str,
    payload: dict[str, object],
) -> dict[str, object]:
    """POST one JSON payload and decode the JSON response."""

    request = urllib.request.Request(
        f"{API_BASE_URL}{path}",
        data=json.dumps(
            payload
        ).encode("utf-8"),
        headers={
            "Content-Type": (
                "application/json"
            ),
        },
        method="POST",
    )

    with urllib.request.urlopen(
        request,
        timeout=30,
    ) as response:
        if response.status != 200:
            raise RuntimeError(
                f"POST {path} returned "
                f"HTTP {response.status}"
            )

        return json.loads(
            response.read()
        )


def get_text(
    url: str,
) -> str:
    """GET one text endpoint."""

    request = urllib.request.Request(
        url,
        method="GET",
    )

    with urllib.request.urlopen(
        request,
        timeout=30,
    ) as response:
        if response.status != 200:
            raise RuntimeError(
                f"GET {url} returned "
                f"HTTP {response.status}"
            )

        return (
            response
            .read()
            .decode("utf-8")
            .strip()
        )


def validate_health() -> None:
    """Require healthy API and Streamlit processes."""

    health = get_json(
        "/health"
    )

    if health != {
        "status": "ok",
        "model_loaded": True,
    }:
        raise AssertionError(
            "Unexpected API health response: "
            f"{health!r}"
        )

    dashboard_health = get_text(
        DASHBOARD_HEALTH_URL
    )

    if dashboard_health != "ok":
        raise AssertionError(
            "Unexpected dashboard health response: "
            f"{dashboard_health!r}"
        )

    print(
        "health checks: ok",
        flush=True,
    )


def validate_model_info() -> None:
    """Require the deployed API to expose the frozen inference contract."""

    info = get_json(
        "/model-info"
    )

    expected = {
        "baseline_model_version": (
            FROZEN_BASELINE_MODEL_VERSION
        ),
        "policy_version": (
            FROZEN_POLICY_VERSION
        ),
        "calibration_method": (
            FROZEN_CALIBRATION_METHOD
        ),
        "review_threshold": (
            FROZEN_REVIEW_THRESHOLD
        ),
        "block_threshold": (
            FROZEN_BLOCK_THRESHOLD
        ),
        "explanation_version": (
            FROZEN_EXPLANATION_VERSION
        ),
        "reason_code_version": (
            FROZEN_REASON_CODE_VERSION
        ),
        "policy_artifact_sha256": (
            FROZEN_POLICY_SHA256
        ),
        "feature_count": 63,
        "categorical_feature_count": 29,
        "numerical_feature_count": 34,
    }

    for key, expected_value in (
        expected.items()
    ):
        actual_value = info.get(
            key
        )

        if actual_value != expected_value:
            raise AssertionError(
                f"Unexpected model-info value "
                f"for {key!r}: "
                f"expected {expected_value!r}, "
                f"got {actual_value!r}"
            )

    print(
        "frozen model-info: ok",
        flush=True,
    )


def build_features() -> dict[str, object]:
    """Build one complete deterministic current-contract feature payload."""

    if str(ROOT) not in sys.path:
        sys.path.insert(
            0,
            str(ROOT),
        )

    from src.data_processing import (
        AMOUNT_COLUMN,
    )
    from src.features import (
        CATEGORICAL_FEATURES,
        FEATURE_COLUMNS,
    )

    categorical = set(
        CATEGORICAL_FEATURES
    )

    features: dict[
        str,
        object,
    ] = {
        feature_name: (
            None
            if feature_name
            in categorical
            else 0.0
        )
        for feature_name
        in FEATURE_COLUMNS
    }

    features[
        AMOUNT_COLUMN
    ] = 125.50

    return features


def validate_predictions() -> None:
    """Persist string and integer transaction IDs through FastAPI."""

    features = build_features()

    for transaction_id in (
        STRING_TRANSACTION_ID,
        INTEGER_TRANSACTION_ID,
    ):
        response = post_json(
            "/predict",
            {
                "transaction_id": (
                    transaction_id
                ),
                "features": features,
            },
        )

        if (
            response.get(
                "transaction_id"
            )
            != transaction_id
        ):
            raise AssertionError(
                "Prediction response changed "
                "transaction-ID type or value: "
                f"{response!r}"
            )

        expected_provenance = {
            "model_version": (
                FROZEN_BASELINE_MODEL_VERSION
            ),
            "policy_version": (
                FROZEN_POLICY_VERSION
            ),
            "explanation_version": (
                FROZEN_EXPLANATION_VERSION
            ),
            "reason_code_version": (
                FROZEN_REASON_CODE_VERSION
            ),
        }

        for key, expected_value in (
            expected_provenance.items()
        ):
            actual_value = response.get(
                key
            )

            if actual_value != expected_value:
                raise AssertionError(
                    "Prediction response did not "
                    "preserve frozen provenance "
                    f"for {key!r}: "
                    f"expected {expected_value!r}, "
                    f"got {actual_value!r}"
                )

    print(
        "PostgreSQL prediction persistence: ok",
        flush=True,
    )


def validate_label_backfill() -> None:
    """Exercise typed label lookup against PostgreSQL."""

    label_script = f'''
from api.logging_db import create_prediction_store
from api.prediction_labels import (
    PredictionLabelUpdate,
    record_prediction_labels,
)

store = create_prediction_store()

try:
    results = record_prediction_labels(
        store,
        [
            PredictionLabelUpdate(
                transaction_id={STRING_TRANSACTION_ID!r},
                actual_label=0,
            ),
            PredictionLabelUpdate(
                transaction_id={INTEGER_TRANSACTION_ID!r},
                actual_label=1,
            ),
        ],
    )

    observed = [
        (
            type(result.transaction_id).__name__,
            result.transaction_id,
            result.actual_label,
            result.matched_events,
            result.updated_events,
        )
        for result in results
    ]

    expected = [
        (
            "str",
            {STRING_TRANSACTION_ID!r},
            0,
            1,
            1,
        ),
        (
            "int",
            {INTEGER_TRANSACTION_ID!r},
            1,
            1,
            1,
        ),
    ]

    if observed != expected:
        raise AssertionError(
            f"Unexpected label results: {{observed!r}}"
        )

    print("typed PostgreSQL label backfill: ok")

finally:
    store.dispose()
'''

    compose(
        "exec",
        "-T",
        "api",
        "python",
        "-",
        input_text=label_script,
    )


def persisted_rows() -> list[str]:
    """Return compact persisted rows directly from PostgreSQL."""

    result = compose(
        "exec",
        "-T",
        "postgres",
        "sh",
        "-c",
        (
            'psql -U "$POSTGRES_USER" '
            '-d "$POSTGRES_DB" '
            '-At -F "|" '
            '-c "SELECT '
            "transaction_id::text, "
            "actual_label "
            "FROM prediction_events "
            'ORDER BY prediction_id;"'
        ),
        capture_output=True,
    )

    return [
        line.strip()
        for line in (
            result.stdout
            .splitlines()
        )
        if line.strip()
    ]


def validate_persisted_rows() -> None:
    """Require exact typed IDs and labels in PostgreSQL."""

    expected = [
        f'"{STRING_TRANSACTION_ID}"|0',
        f"{INTEGER_TRANSACTION_ID}|1",
    ]

    observed = persisted_rows()

    if observed != expected:
        raise AssertionError(
            "Unexpected PostgreSQL rows: "
            f"expected {expected!r}, "
            f"got {observed!r}"
        )

    print(
        "direct PostgreSQL verification: ok",
        flush=True,
    )


def main() -> None:
    """Run the complete isolated deployment smoke workflow."""

    compose(
        "config",
        "--quiet",
    )

    compose(
        "down",
        "--volumes",
        "--remove-orphans",
    )

    try:
        compose(
            "up",
            "--build",
            "--detach",
            "--wait",
            "--wait-timeout",
            "120",
        )

        validate_health()
        validate_model_info()
        validate_predictions()
        validate_label_backfill()
        validate_persisted_rows()

        print(
            "recreating containers without "
            "deleting PostgreSQL volume...",
            flush=True,
        )

        compose(
            "down",
            "--remove-orphans",
        )

        compose(
            "up",
            "--detach",
            "--wait",
            "--wait-timeout",
            "120",
        )

        validate_health()
        validate_model_info()
        validate_persisted_rows()

        print(
            "deployment smoke: PASS",
            flush=True,
        )

    except Exception:
        print(
            "deployment smoke: FAIL",
            file=sys.stderr,
            flush=True,
        )

        try:
            compose(
                "ps",
            )
            compose(
                "logs",
                "--no-color",
                "--tail",
                "100",
            )
        except Exception:
            pass

        raise

    finally:
        compose(
            "down",
            "--volumes",
            "--remove-orphans",
        )


if __name__ == "__main__":
    main()
