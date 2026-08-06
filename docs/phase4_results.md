# Phase 4 SHAP Explanations and Reason Codes

## Status

Phase 4 is complete.

The frozen `baseline-v1` LightGBM model and `calibrated-policy-v1` policy now
support deterministic, validated model-signal explanations.

Phase 4 did not retrain the model, refit the calibrator, change the feature
contract, or revise either policy threshold.

Completed on 6 August 2026.

## Frozen configuration

| Field | Value |
|---|---|
| Model type | LightGBM binary classifier |
| Model version | `baseline-v1` |
| Feature count | 63 |
| Categorical features | 29 |
| Numerical features | 34 |
| Policy version | `calibrated-policy-v1` |
| Calibration method | Sigmoid |
| Review threshold | `0.16255069862369795` |
| Block threshold | `0.8509223095305902` |
| Explanation version | `shap-explanation-v1` |
| Reason-code version | `reason-codes-v1` |
| Reconstruction tolerance | `1e-8` |

The frozen policy SHA-256 remains:

```text
5d53f23719ae891ecc24585393585765aa7fc0900ab38f95e37f59c18fe6c90f
```

## Explanation workflow

```text
raw transaction features
    -> frozen categorical encoder
    -> frozen LightGBM raw model score
    -> frozen sigmoid calibration
    -> ALLOW / REVIEW / BLOCK decision
    -> native LightGBM TreeSHAP contributions
    -> raw-margin reconstruction validation
    -> raw-score reconstruction validation
    -> deterministic contribution ranking
    -> stable analyst-facing reason codes
```

The explanation stage is observational. It does not feed values back into
model scoring, calibration, or policy decisions.

## Implemented components

- Immutable contribution and explanation contracts
- `OBSERVED`, `MISSING`, and `UNKNOWN_CATEGORY` input states
- Native LightGBM contribution extraction
- Raw-margin and raw-score reconstruction checks
- Stable feature groups and versioned reason codes
- Deterministic contribution ranking and reason deduplication
- Explanation-enabled calibrated-policy inference
- Batch and individual-row parity
- Policy reload parity
- Frozen artifact and model-state immutability checks

## Reason-code contract

Reason codes use:

```text
<FEATURE_GROUP>_<VALUE_STATE>_<DIRECTION>
```

Messages describe model-score influence and avoid causal fraud claims.

## Determinism and parity

Repeated calls for identical inputs produce identical contributions, ranks,
reason codes, messages, raw scores, calibrated probabilities, and decisions.
A row explained alone produces the same result as that row inside a batch.

## Automated validation

| Test area | Tests |
|---|---:|
| Explanation contracts and ranking | 34 |
| Feature groups and reason codes | 54 |
| Feature-value states | 13 |
| Native LightGBM contributions | 8 |
| SHAP reconstruction | 12 |
| Explanation assembly | 12 |
| Policy integration | 10 |
| Frozen artifact integration | 4 |
| **Focused Phase 4 total** | **147** |

The final completion workflow also requires the complete repository test suite
to pass.

## Interpretation

SHAP values explain the frozen LightGBM raw-margin signals. They do not directly
decompose the calibrated probability and must not be interpreted as causal
evidence of fraud.

## Limitations

- Explanations are associative model attributions, not causal findings.
- SHAP contributions decompose the raw margin, not calibrated probability.
- Raw category and identifier values are not exposed in reason messages.
- No counterfactual recommendations are provided.
- Production explanation latency has not been benchmarked.
- The explanation contract is not yet exposed through the production API.
- No explanation monitoring workflow exists yet.

## Phase 4 conclusion

Phase 4 delivered deterministic native TreeSHAP explanations, stable reason
codes, reconstruction validation, calibrated-policy integration, batch parity,
reload parity, and frozen-artifact immutability checks.

The next phase is production API prediction integration.
