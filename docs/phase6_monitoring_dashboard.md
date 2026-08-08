# Phase 6 — Monitoring and Threshold-Simulation Dashboard

## Status

Phase 6 is complete.

This phase adds local prediction persistence, outcome-label support,
read-only monitoring, and a Streamlit threshold simulator around the
frozen Phase 5 inference system.

Phase 6 does not retrain, recalibrate, retune, overwrite, or promote
any model or policy artifact.

## Frozen Contract

| Component | Value |
|---|---|
| Baseline model | `baseline-v1` |
| Policy | `calibrated-policy-v1` |
| Calibration | `sigmoid` |
| REVIEW threshold | `0.16255069862369795` |
| BLOCK threshold | `0.8509223095305902` |
| Explanation version | `shap-explanation-v1` |
| Reason-code version | `reason-codes-v1` |
| Policy SHA-256 | `5d53f23719ae891ecc24585393585765aa7fc0900ab38f95e37f59c18fe6c90f` |

Frozen artifact:

`models/payguard_calibrated_policy.joblib`

The artifact remained unchanged throughout Phase 6.

## Architecture

Phase 6 surrounds the frozen Phase 5 inference pipeline:

HTTP request
→ validation
→ frozen model/calibrator/policy
→ PredictionResponse
→ prediction-event persistence
→ monitoring / threshold simulation

Monitoring observes persisted production-policy decisions.

Threshold simulation applies temporary candidate thresholds to persisted
calibrated probabilities without changing production policy state.

## Prediction Persistence

Implemented in:

- `api/logging_db.py`
- `api/persistence.py`
- `api/main.py`

Default local database:

`sqlite:///./predictions.db`

Each successful prediction stores an append-only prediction event.

Stored fields include:

- prediction ID
- transaction ID
- raw model score
- calibrated probability
- persisted ALLOW / REVIEW / BLOCK decision
- transaction amount when available
- model version
- policy version
- calibration method
- frozen REVIEW and BLOCK thresholds
- frozen policy SHA-256
- explanation version
- reason-code version
- reason codes
- scoring timestamp
- optional ground-truth label
- optional label timestamp

Repeated scoring of the same transaction ID creates separate prediction
events. Earlier events are not overwritten.

Single prediction persistence follows:

validate → infer → persist → return

Batch prediction persistence follows:

validate → infer ordered batch → atomically persist batch → return

Validation or inference failure creates no prediction event.

Persistence failure does not produce a successful HTTP 200 prediction.

## Outcome Labels

Implemented in:

`api/prediction_labels.py`

Ground truth uses:

- `0` = legitimate
- `1` = fraud

Label updates:

- preserve integer-versus-string transaction-ID distinction
- reject invalid labels
- require UTC timestamps
- reject unknown transactions
- reject conflicting relabels
- allow idempotent replay of the same label
- preserve the original label timestamp
- update repeated persisted events for the same transaction
- apply batches atomically

No public label-management HTTP endpoint was added in Phase 6.

## Monitoring

Implemented in:

`dashboard/monitoring.py`

Monitoring uses persisted API decisions as the authoritative historical
decisions.

It reports:

- total event volume
- ALLOW count and rate
- REVIEW count and rate
- BLOCK count and rate
- intervention count and rate
- label coverage
- calibrated-score statistics
- chronological score observations
- reason-code frequencies
- recent REVIEW/BLOCK events
- persisted policy provenance

Monitoring does not infer fraud performance from unlabeled events.

Without ground-truth labels, the system does not claim:

- fraud recall
- fraud precision
- fraud capture
- fraud prevalence
- economic savings

## Threshold Simulation

Implemented in:

`dashboard/simulation.py`

Candidate decisions reuse:

`src.policy.assign_policy_decisions()`

Decision semantics remain:

- probability below REVIEW threshold → ALLOW
- probability at or above REVIEW threshold but below BLOCK → REVIEW
- probability at or above BLOCK threshold → BLOCK

Candidate thresholds are temporary only.

Phase 6 provides no operation to:

- save thresholds
- apply thresholds
- promote thresholds
- overwrite the frozen artifact
- modify persisted decisions
- search for optimized thresholds

Phase 6 does not call:

`search_policy_thresholds()`

## Workload Simulation

All persisted events can support workload simulation.

The simulator compares frozen and candidate:

- ALLOW count and rate
- REVIEW count and rate
- BLOCK count and rate
- intervention count and rate
- changed-decision count
- changed-decision rate
- decision transitions

Operational workload constraints are evaluated without requiring labels:

- maximum review rate
- maximum block rate
- maximum intervention rate

## Labeled Fraud and Economic Evaluation

Fraud and economic metrics are shown only when supported by eligible
ground-truth data.

Economic evaluation requires:

- a binary ground-truth label
- a transaction amount
- both legitimate and fraud examples in the eligible labeled subset

The simulator reuses:

- `src.policy.evaluate_policy()`
- `src.policy.evaluate_policy_constraints()`

Frozen development economics:

| Assumption | Value |
|---|---:|
| Review cost | `2.00` |
| Legitimate block cost | `5.00` |
| Fraud loss multiplier | `1.00` |
| Review fraud capture rate | `50%` |

Frozen operational constraints:

| Constraint | Value |
|---|---:|
| Maximum review rate | `5%` |
| Maximum block rate | `1%` |
| Maximum intervention rate | `6%` |

These economic values are development assumptions, not validated
production costs or measured production savings.

## Streamlit Dashboard

Implemented in:

- `dashboard/app.py`
- `dashboard/monitoring_view.py`
- `dashboard/simulation_view.py`

The dashboard contains two views.

### Monitoring

Displays:

- frozen policy provenance
- prediction volume
- actual decision distribution
- intervention rate
- calibrated-score monitoring
- reason-code frequency
- recent REVIEW/BLOCK events
- outcome-label coverage

### Threshold Simulator

Displays:

- frozen thresholds
- temporary candidate thresholds
- workload comparison
- changed-decision rate
- decision transitions
- operational constraint feasibility
- labeled fraud/economic metrics when supported

There is no save, apply, or promote control.

## Running Locally

Start the API:

`python -m uvicorn api.main:app --reload`

Start the dashboard in another terminal:

`python -m streamlit run dashboard/app.py`

Both use the configured:

`settings.database_url`

The default local SQLite database is:

`predictions.db`

## HTTP Surface

Phase 6 does not add monitoring or simulation HTTP endpoints.

The public API remains:

| Method | Path |
|---|---|
| GET | `/health` |
| GET | `/model-info` |
| POST | `/predict` |
| POST | `/batch-predict` |

Prediction persistence occurs behind successful prediction requests.

There is no `/recent-predictions` endpoint.

There is no `/threshold-simulation` endpoint.

## Verification

Before the Phase 6 documentation commit, the full suite completed with:

`447 passed`

Focused verification covers:

- prediction-store behavior
- atomic persistence
- API persistence
- transaction-ID typing
- outcome labels
- monitoring aggregation
- read-only monitoring
- threshold boundary semantics
- frozen-threshold parity
- decision transitions
- operational constraints
- label-gated fraud/economic evaluation
- Monitoring Streamlit smoke tests
- Threshold Simulator Streamlit smoke tests
- dashboard read-only behavior

Frozen artifact SHA-256 remained:

`5d53f23719ae891ecc24585393585765aa7fc0900ab38f95e37f59c18fe6c90f`

## Phase 6 Implementation Commits

- `42cb703` — `refactor(db): define Phase 6 prediction event store`
- `e3c5456` — `feat(api): persist successful prediction events`
- `90504ff` — `feat(db): support prediction outcome labels`
- `4933e62` — `feat(monitoring): add prediction monitoring queries`
- `3cbb190` — `feat(simulation): add read-only threshold scenarios`
- `0e3882d` — `feat(dashboard): implement prediction monitoring view`
- `95214e6` — `feat(dashboard): implement threshold simulator`

## Remaining Limitations

Phase 6 is a local monitoring and analysis layer, not a production fraud
operations platform.

Remaining work includes:

- automated ground-truth ingestion
- monitoring alerts
- database migrations
- production PostgreSQL validation
- authentication and authorization
- production deployment
- production traffic validation
- production cost validation
- automated retraining
- formal policy approval and model-risk governance

Phase 7 can focus on deployment without changing the frozen Phase 6
monitoring and simulation contract.
