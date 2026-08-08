# RiskFlow PayGuard

## Payment Fraud Risk Scoring Platform

RiskFlow PayGuard is an end-to-end payment fraud-risk engineering project built around the IEEE-CIS Fraud Detection dataset.

The project is being developed as a deployable fintech-style product rather than only a notebook-based machine-learning exercise. Its target workflow converts raw payment data into a fraud-risk score, an operational `ALLOW`, `REVIEW`, or `BLOCK` decision, an explanation, and monitoring outputs.

## Current Status

The data, baseline-model, probability-calibration, drift-diagnostic, decision-policy, deterministic explanation, FastAPI inference, prediction persistence, monitoring, and read-only threshold-simulation foundations are complete.

| Phase | Status | Deliverable |
|---|---|---|
| Phase 1 | Complete | Local data workflow, EDA, feature engineering, and chronological model datasets |
| Phase 2 | Complete | Reproducible LightGBM baseline, fraud-focused evaluation, and versioned model bundle |
| Phase 3 | Complete | Probability calibration, drift diagnostics, policy optimization, and versioned policy bundle |
| Phase 4 | Complete | Native LightGBM TreeSHAP, deterministic contributions, reason codes, and artifact immutability checks |
| Phase 5 | Complete | Strict FastAPI inference, frozen-policy loading, model metadata, single/batch prediction, parity, and immutability checks |
| Phase 6 | Complete | Prediction persistence, outcome labeling, read-only monitoring, Streamlit monitoring, and threshold simulation |
| Phase 7 | Complete | Docker image, PostgreSQL Compose stack, health checks, and deployment smoke validation |
| Phase 8+ | Planned | Public cloud release, monitoring alerts, governance, and portfolio polish |

### Implemented

- IEEE-CIS local data workflow with raw data excluded from Git
- Chronological training, validation, and test splits
- Validated 63-feature model contract
- 29 categorical and 34 numerical features
- Training-only categorical vocabularies
- Missing-category and unseen-category handling
- LightGBM binary classifier with class weighting
- Validation-based early stopping
- PR-AUC, ROC-AUC, log loss, and Brier score
- Reference-threshold confusion metrics
- Fraud recall and precision at fixed review capacities
- Fraudulent transaction-amount capture metrics
- Versioned and atomically persisted joblib model bundle
- Sigmoid probability calibration
- Calibration-fit and policy-selection validation partitions
- Score and categorical drift diagnostics
- Explicit fraud-policy costs and operational constraints
- Deterministic `ALLOW`, `REVIEW`, and `BLOCK` threshold optimization
- Versioned calibrated policy bundle
- Artifact overwrite protection and post-load validation
- Raw-feature calibrated-policy inference
- Native LightGBM TreeSHAP contribution extraction
- Raw-margin and raw-score reconstruction validation
- Deterministic top positive and negative contributions
- Stable versioned analyst reason codes
- Observed, missing, and unknown-category explanation states
- Explanation-enabled calibrated-policy inference
- Batch and individual-row explanation parity
- Frozen model and policy artifact immutability checks
- Reloaded-bundle inference tests
- Automated pytest coverage
- Fail-closed FastAPI lifespan loading of the frozen calibrated policy
- Frozen policy SHA-256 verification before deserialization
- Strict Pydantic v2 transaction and batch schemas
- Complete 63-feature API request contract
- `GET /health`
- `GET /model-info`
- `POST /predict`
- `POST /batch-predict`
- Direct API versus frozen inference parity
- Single-versus-batch prediction parity
- Ordered batch identifier and row preservation
- Standard HTTP 422 validation behavior
- API-level encoder and policy-artifact immutability checks
- Deterministic repeated API responses
- SQLAlchemy prediction-event persistence
- Local SQLite prediction-event storage
- Atomic single and batch prediction persistence
- Repeated transaction-event history without overwriting earlier scores
- Frozen policy provenance stored with every prediction event
- Ground-truth outcome-label backfill
- Conflict-safe and idempotent label updates
- Read-only monitoring of persisted ALLOW, REVIEW, and BLOCK decisions
- Label-coverage reporting without unsupported fraud claims
- Streamlit Monitoring view
- Temporary read-only threshold scenarios
- Frozen-versus-candidate workload and decision-transition analysis
- Label-gated fraud and development-economics comparison
- Streamlit Threshold Simulator with no save, apply, or promote workflow

### Not Yet Implemented

- Automated monitoring alerts
- Automated ground-truth ingestion
- Database migrations and production PostgreSQL validation
- Production cost-assumption validation
- Cloud deployment

The Phase 5 FastAPI inference contract remains frozen. Phase 6 adds local prediction persistence and a read-only Streamlit monitoring/simulation layer around that existing inference system.

---

## Phase 2 Baseline Results

The frozen `baseline-v1` model was trained on the training split, used the validation split for early stopping, and was evaluated once against the untouched chronological test split.

| Metric | Validation | Test |
|---|---:|---:|
| PR-AUC | 0.5772 | 0.4946 |
| ROC-AUC | 0.9128 | 0.8808 |
| Log loss | 0.1116 | 0.1300 |
| Brier score | 0.0290 | 0.0336 |
| Fraud prevalence | 3.43% | 3.48% |

At a review capacity of the top 1% of test transactions, the model achieved:

- 24.10% fraud recall
- 83.86% review precision
- 14.16% fraudulent amount capture

The test PR-AUC is approximately 14.2 times the underlying test fraud prevalence. The temporal decline from validation to test indicates that drift and stability require further investigation.

The baseline is accepted as a technical benchmark, not as a calibrated or production-ready fraud policy.

See the complete results and implementation plan:

- [`docs/phase2_baseline_results.md`](docs/phase2_baseline_results.md)
- [`docs/phase2_baseline_model.md`](docs/phase2_baseline_model.md)

---


## Phase 3 Calibration and Policy Results

The frozen `calibrated-policy-v1` artifact embeds `baseline-v1`, a
sigmoid calibrator, and deterministic fraud-policy thresholds.

Selected thresholds:

- `REVIEW`: probability at or above `0.16255069862369795`
- `BLOCK`: probability at or above `0.8509223095305902`

Final one-time chronological test results:

| Metric | Result |
|---|---:|
| PR-AUC | 0.4946 |
| ROC-AUC | 0.8808 |
| Calibrated log loss | 0.1014 |
| Calibrated Brier score | 0.0238 |
| Review rate | 4.44% |
| Block rate | 0.00% |
| Review precision | 42.49% |
| Fraud intervention recall | 54.23% |
| Fraud amount capture | 21.13% |
| Modeled cost reduction versus all-allow | 19.45% |

Calibration improved test log loss from `0.1300` to `0.1014` and Brier
score from `0.0336` to `0.0238`, while monotonic sigmoid calibration
preserved ranking performance.

The frozen policy satisfied all configured operational constraints.
No test transaction exceeded the frozen block threshold, so the
test-period policy operated as an `ALLOW` or `REVIEW` policy.

Economic results depend on explicit development assumptions and are not
measured production savings.

See:

- [`docs/phase3_results.md`](docs/phase3_results.md)
- [`docs/phase3_calibration_thresholds.md`](docs/phase3_calibration_thresholds.md)

---

## Phase 4 Explanation Results

The frozen `baseline-v1` LightGBM model now supports native TreeSHAP
explanations integrated with `calibrated-policy-v1`.

TreeSHAP contributions reconstruct the raw model margin within an absolute
tolerance of `1e-8`. Explanation-enabled inference preserves the existing raw
score, calibrated probability, decision, and row order.

The explanation layer is read-only. Frozen model trees, categorical
vocabularies, calibration parameters, policy thresholds, and artifact bytes
remain unchanged.

Reason codes describe model signals and must not be interpreted as causal
evidence of fraud.

See:

- [`docs/phase4_results.md`](docs/phase4_results.md)
- [`docs/phase4_explanations_reason_codes.md`](docs/phase4_explanations_reason_codes.md)

---

## Dataset

RiskFlow PayGuard uses the IEEE-CIS Fraud Detection dataset from Kaggle.

Raw, processed, and model-binary files are intentionally excluded from Git.

Expected local structure:

```text
data/
├── raw/
├── processed/
└── sample_payloads/
```

Processed labeled datasets:

```text
data/processed/train.parquet
data/processed/validation.parquet
data/processed/test.parquet
```

Supporting metadata:

```text
data/processed/feature_metadata.csv
data/processed/dataset_manifest.json
```

Chronological split sizes:

| Split | Rows |
|---|---:|
| Training | 413,378 |
| Validation | 88,581 |
| Test | 88,581 |

See [`data/README.md`](data/README.md) for the local data setup instructions.

---

## Modeling Design

The Phase 2 baseline uses:

- `LGBMClassifier`
- Binary classification objective
- Learning rate of `0.03`
- Maximum of `2,000` estimators
- Validation-based early stopping
- Positive-class weighting from the training distribution
- Deterministic CPU configuration
- Native LightGBM numerical missing-value handling
- Explicitly identified categorical features

The fitted model stopped at iteration `1,454`.

Categorical preprocessing is learned from the training split only. Validation, test, and future inference data cannot expand the learned vocabularies.

The versioned model bundle contains:

- Fitted LightGBM classifier
- Categorical encoder
- Ordered feature names
- Categorical and numerical feature groups
- Category vocabularies
- Model configuration
- Positive-class weight
- Best iteration
- Validation metrics
- Final test metrics
- Source dataset manifest
- Model version and creation timestamp

Generated model artifacts are written under `models/` and remain excluded from Git.

---

## Stack

### Implemented

- Python
- pandas
- NumPy
- PyArrow
- scikit-learn
- LightGBM
- joblib
- pytest
- FastAPI
- Pydantic v2
- Uvicorn
- SHAP / native LightGBM TreeSHAP explanations
- SQLAlchemy prediction-event persistence
- SQLite for local prediction logs
- Streamlit monitoring and threshold simulation

### Planned Next Product Layer

- Database migration tooling before future schema changes
- Docker
- Cloud deployment
- Monitoring alerts

---

## Local Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

LightGBM requires the OpenMP runtime on macOS. Install it through Homebrew when needed:

```bash
brew install libomp
```

### 3. Prepare the data

Follow [`data/README.md`](data/README.md).

Raw Kaggle files should be placed under:

```text
data/raw/
```

They must not be committed.

### 4. Run the test suite

```bash
python -m pytest
```

### 5. Run the Phase 5 API locally

Start the local FastAPI application with:

    python -m uvicorn api.main:app --reload

The application loads and verifies the frozen calibrated policy during
lifespan startup. A missing, replaced, or incompatible policy artifact causes
startup to fail rather than silently falling back to another model.

Interactive OpenAPI documentation is available from the local FastAPI `/docs`
route while the service is running.

### 6. Run the Phase 6 dashboard locally

In a second terminal, start Streamlit with:

    python -m streamlit run dashboard/app.py

The API and dashboard use the same configured prediction database. The default
local database URL is `sqlite:///./predictions.db`.

Successful `/predict` and `/batch-predict` requests populate the monitoring
store. The Threshold Simulator uses temporary candidate thresholds only and
does not modify the frozen production policy or persisted production decisions.

### 7. Train a local development model

Use `--skip-test-evaluation` during development to avoid repeatedly inspecting the frozen chronological test split:

```bash
python -m src.train \
  --model-output models/payguard_baseline_local.joblib \
  --model-version baseline-local-v1 \
  --n-jobs 4 \
  --skip-test-evaluation
```

The published `baseline-v1` test results are already recorded. Further implementation decisions should not be tuned against that test result.

### 8. Load a saved bundle

```python
from src.model_bundle import load_model_bundle

bundle = load_model_bundle(
    "models/payguard_baseline_local.joblib"
)

print(bundle.model_version)
print(bundle.best_iteration)
print(bundle.feature_columns)
```

Compatible raw feature frames can be scored with:

```python
fraud_probabilities = (
    bundle.predict_fraud_probabilities(features)
)
```

---

## Target Architecture

```text
payment transaction
        |
        v
RiskFlow PayGuard API
        |
        v
feature validation and preprocessing
        |
        v
LightGBM model bundle
        |
        v
fraud-risk score
        |
        v
decision engine
   |       |       |
 ALLOW   REVIEW   BLOCK
        |
        v
prediction logging
        |
        v
monitoring and threshold dashboard
```

The baseline model bundle, calibrated policy bundle, deterministic explanation layer, Phase 5 API prediction integration, Phase 6 prediction persistence, monitoring, and read-only threshold simulation now exist. Deployment, automated alerts, production database operations, and production governance remain future work.

---

## Implemented Phase 5 API Capabilities

Phase 5 provides a strict and deterministic HTTP interface over the frozen
`calibrated-policy-v1` inference pipeline.

Implemented capabilities include:

- complete Pydantic v2 transaction validation
- frozen policy integrity verification during application startup
- raw fraud-model scoring
- sigmoid-calibrated fraud probabilities
- frozen `ALLOW`, `REVIEW`, and `BLOCK` policy decisions
- deterministic TreeSHAP contribution rankings
- deterministic analyst reason codes
- reconstruction diagnostics
- transaction identifier preservation
- ordered batch prediction
- single-versus-batch parity
- artifact and encoder immutability checks

Implemented endpoints:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Confirm successful API startup and policy loading |
| `GET` | `/model-info` | Return frozen policy and feature-contract metadata |
| `POST` | `/predict` | Score and explain one transaction |
| `POST` | `/batch-predict` | Score and explain an ordered transaction batch |

Prediction logging was intentionally not part of the Phase 5 application surface. Phase 6 now persists successful prediction events behind the existing prediction endpoints and provides monitoring and threshold simulation through the local Streamlit dashboard. `/recent-predictions` and `/threshold-simulation` HTTP endpoints were intentionally not added.

See [`docs/phase5_api_integration.md`](docs/phase5_api_integration.md) for the
complete API contract, implementation record, parity requirements, and final
verification evidence.

---

## Implemented Phase 6 Monitoring and Threshold Simulation

Phase 6 adds a local operational layer around the frozen Phase 5 inference
system without changing how transactions are scored, calibrated, explained,
or assigned under the production policy.

Implemented capabilities include:

- append-only prediction-event persistence
- atomic single and batch persistence
- outcome-label backfill
- read-only monitoring of persisted decisions
- label-coverage reporting
- temporary candidate threshold scenarios
- frozen-versus-candidate workload comparison
- decision-transition analysis
- operational constraint evaluation
- label-gated fraud and development-economics comparison
- Streamlit Monitoring and Threshold Simulator views
- no save, apply, or promote threshold workflow

The simulator reuses the existing policy decision and evaluation functions.
It does not search for optimized thresholds or modify the frozen policy
artifact.

See [`docs/phase6_monitoring_dashboard.md`](docs/phase6_monitoring_dashboard.md)
for the complete Phase 6 persistence, monitoring, simulation, and verification
record.

---

## Implemented Phase 7 Deployment Foundation

Phase 7 packages the existing FastAPI and Streamlit application for
containerized operation without changing the frozen inference contract.

The deployment foundation includes:

- one hardened Python 3.11 application image shared by FastAPI and Streamlit
- build-time frozen-policy integrity validation
- non-root container execution
- PostgreSQL 16 persistence through Docker Compose
- shared PostgreSQL access for API persistence and dashboard reads
- health checks for PostgreSQL, FastAPI, and Streamlit
- named-volume persistence across container recreation
- PostgreSQL-safe typed transaction-ID label lookup
- isolated end-to-end deployment smoke validation

Native local execution continues to default to SQLite. PostgreSQL is the
validated containerized deployment database.

Phase 7 does not claim a public cloud production release.

See [`docs/phase7_deployment.md`](docs/phase7_deployment.md).

---

## Project Structure

```text
api/          FastAPI application, frozen-policy loading, prediction services, persistence, and outcome labeling
dashboard/    Read-only monitoring queries, threshold simulation, and Streamlit views
data/         Local raw, processed, and sample-payload directories
db/           Prediction-log database components
docs/         Architecture, implementation plans, and model results
models/       Local serialized model artifacts
notebooks/    EDA and analytical notebooks
src/          Data processing, preprocessing, evaluation, training, and model bundles
scripts/      Deployment and operational validation utilities
tests/        Automated pytest suite
```

---

## Roadmap

- [x] Phase 1 — Data setup, EDA, feature engineering, and chronological datasets
- [x] Phase 2 — LightGBM baseline, evaluation, and versioned model bundle
- [x] Phase 3 — Probability calibration, drift analysis, and decision thresholds
- [x] Phase 4 — SHAP explanations and reason codes
- [x] Phase 5 — RiskFlow PayGuard API prediction integration
- [x] Phase 6 — Monitoring and threshold-simulation dashboard
- [x] Phase 7 — Docker and PostgreSQL deployment foundation
- [ ] Phase 8 — Portfolio polish and product demonstration

---

## Current Limitations

The calibrated policy should not be treated as a deployed payment
authorization system.

Principal limitations include:

- Development cost assumptions are not validated against production data
- No final-test transaction reached the frozen block threshold
- SHAP reason codes explain model signals but are not causal evidence
- SHAP values decompose the raw margin rather than the calibrated probability
- No dedicated high-value fraud objective
- No public cloud release, authentication, rate limiting, or production traffic validation
- Local prediction persistence is implemented, but there is no automated monitoring-alert engine
- Ground-truth labels require explicit backfill; there is no automated outcome-ingestion pipeline
- SQLite remains the native local default and PostgreSQL is validated through Docker Compose; no schema-migration framework is configured yet
- No automated retraining workflow
- No formal policy approval or model-risk governance process

The recorded `baseline-v1` model and `calibrated-policy-v1` policy should
remain frozen benchmarks for future improvements.

---

## License

MIT
