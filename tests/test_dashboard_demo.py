"""Tests for the Phase 8 API-backed Streamlit Product Demo."""

from __future__ import annotations

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from api.config import (
    FROZEN_BASELINE_MODEL_VERSION,
    FROZEN_EXPLANATION_VERSION,
    FROZEN_POLICY_VERSION,
    FROZEN_REASON_CODE_VERSION,
    settings,
)
from dashboard.demo import (
    DemoAPIError,
    load_demo_scenarios,
    submit_demo_prediction,
)


def _prediction_payload(
    transaction_id: str,
) -> dict[str, object]:
    """Return one valid frozen prediction response for UI tests."""

    return {
        "transaction_id": (
            transaction_id
        ),
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
        "raw_model_score": 0.01,
        "calibrated_probability": 0.02,
        "decision": "ALLOW",
        "top_positive_contributions": [
            {
                "feature": "C14",
                "feature_index": 25,
                "feature_group": "count_C",
                "direction": (
                    "INCREASES_SCORE"
                ),
                "shap_value_raw": 0.20,
                "absolute_shap_value_raw": 0.20,
                "value_state": "OBSERVED",
                "rank": 1,
            }
        ],
        "top_negative_contributions": [
            {
                "feature": "D1",
                "feature_index": 27,
                "feature_group": "delta_D",
                "direction": (
                    "DECREASES_SCORE"
                ),
                "shap_value_raw": -0.10,
                "absolute_shap_value_raw": 0.10,
                "value_state": "OBSERVED",
                "rank": 1,
            }
        ],
        "reason_codes": [
            (
                "COUNT_AGGREGATE_"
                "OBSERVED_INCREASES_SCORE"
            )
        ],
        "reasons": [
            {
                "code": (
                    "COUNT_AGGREGATE_"
                    "OBSERVED_INCREASES_SCORE"
                ),
                "message": (
                    "Observed transaction-count "
                    "signals increased model score."
                ),
            }
        ],
        "reconstruction": {
            "raw_model_margin": -4.0,
            "expected_value_raw": -4.1,
            "shap_sum_raw": 0.1,
            "reconstructed_raw_margin": -4.0,
            "reconstructed_raw_model_score": (
                0.01798620996209156
            ),
            "margin_reconstruction_error": 0.0,
            "score_reconstruction_error": 0.0,
        },
    }


class _FakeResponse:
    """Minimal successful httpx-style response."""

    status_code = 200

    def __init__(
        self,
        payload: dict[str, object],
    ) -> None:
        self._payload = payload

    def json(
        self,
    ) -> dict[str, object]:
        return self._payload


def test_demo_scenarios_load_committed_batch() -> None:
    """Expose the three validated committed synthetic scenarios."""

    scenarios = (
        load_demo_scenarios()
    )

    assert [
        scenario.transaction.transaction_id
        for scenario
        in scenarios
    ] == [
        "demo-everyday-001",
        "demo-higher-value-002",
        "demo-mobile-identity-004",
    ]

    assert [
        scenario.label
        for scenario
        in scenarios
    ] == [
        "Everyday purchase",
        "Higher-value purchase",
        "Mobile identity-rich purchase",
    ]

    assert all(
        len(
            scenario
            .transaction
            .features
            .model_dump()
        )
        == 63
        for scenario
        in scenarios
    )


def test_submit_demo_prediction_uses_real_predict_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Submit the committed transaction to /predict without local inference."""

    scenario = (
        load_demo_scenarios()[
            0
        ]
    )

    captured: dict[
        str,
        object,
    ] = {}

    def fake_post(
        url: str,
        *,
        json: dict[str, object],
        timeout: float,
    ) -> _FakeResponse:
        captured[
            "url"
        ] = url
        captured[
            "json"
        ] = json
        captured[
            "timeout"
        ] = timeout

        return _FakeResponse(
            _prediction_payload(
                "demo-everyday-001"
            )
        )

    monkeypatch.setattr(
        "dashboard.demo.httpx.post",
        fake_post,
    )

    prediction = (
        submit_demo_prediction(
            scenario.transaction,
            "http://payguard.test/",
            timeout_seconds=3.0,
        )
    )

    assert (
        captured["url"]
        == "http://payguard.test/predict"
    )
    assert (
        captured["json"]
        == scenario
        .transaction
        .model_dump(
            mode="json"
        )
    )
    assert (
        captured["timeout"]
        == 3.0
    )

    assert (
        prediction.transaction_id
        == "demo-everyday-001"
    )
    assert (
        prediction.decision
        == "ALLOW"
    )


def test_submit_demo_prediction_fails_closed_when_api_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never fall back to local model scoring when FastAPI is unavailable."""

    scenario = (
        load_demo_scenarios()[
            0
        ]
    )

    def fail_post(
        url: str,
        *,
        json: dict[str, object],
        timeout: float,
    ) -> None:
        del json
        del timeout

        request = httpx.Request(
            "POST",
            url,
        )

        raise httpx.ConnectError(
            "offline",
            request=request,
        )

    monkeypatch.setattr(
        "dashboard.demo.httpx.post",
        fail_post,
    )

    with pytest.raises(
        DemoAPIError,
        match=(
            "No local scoring fallback "
            "was used"
        ),
    ):
        submit_demo_prediction(
            scenario.transaction,
            "http://payguard.test",
        )


def test_product_demo_default_and_scoring_smoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Render the default demo and display a mocked live API prediction."""

    captured: dict[
        str,
        object,
    ] = {}

    def fake_post(
        url: str,
        *,
        json: dict[str, object],
        timeout: float,
    ) -> _FakeResponse:
        captured[
            "url"
        ] = url
        captured[
            "json"
        ] = json
        captured[
            "timeout"
        ] = timeout

        return _FakeResponse(
            _prediction_payload(
                "demo-everyday-001"
            )
        )

    monkeypatch.setattr(
        "dashboard.demo.httpx.post",
        fake_post,
    )
    monkeypatch.setattr(
        settings,
        "payguard_api_url",
        "http://payguard.test",
    )

    app_test = (
        AppTest.from_file(
            "dashboard/app.py"
        )
        .run(
            timeout=10
        )
    )

    assert (
        len(
            app_test.exception
        )
        == 0
    )

    assert (
        app_test.header[
            0
        ].value
        == "Product Demo"
    )

    assert (
        app_test.selectbox[
            0
        ].value
        == "Everyday purchase"
    )

    assert (
        app_test.button[
            0
        ].label
        == "Score transaction"
    )

    app_test.button[
        0
    ].click()

    app_test = (
        app_test.run(
            timeout=10
        )
    )

    assert (
        len(
            app_test.exception
        )
        == 0
    )

    metrics = {
        metric.label: metric.value
        for metric
        in app_test.metric
    }

    assert (
        metrics["Decision"]
        == "ALLOW"
    )
    assert (
        metrics[
            "Calibrated risk probability"
        ]
        == "2.0%"
    )

    assert any(
        "Scored through FastAPI"
        in success.value
        for success
        in app_test.success
    )

    assert (
        captured["url"]
        == "http://payguard.test/predict"
    )

    request_payload = (
        captured["json"]
    )
    assert isinstance(
        request_payload,
        dict,
    )
    assert (
        request_payload[
            "transaction_id"
        ]
        == "demo-everyday-001"
    )
