# RiskFlow PayGuard

## Payment Fraud Risk Scoring Platform

RiskFlow PayGuard is an end-to-end payment fraud-risk engineering project built around the IEEE-CIS Fraud Detection dataset.

The project is being developed as a deployable fintech-style product rather than only a notebook-based machine-learning exercise. Its target workflow converts raw payment data into a fraud-risk score, an operational `ALLOW`, `REVIEW`, or `BLOCK` decision, an explanation, and monitoring outputs.

## Current Status

The data, baseline-model, probability-calibration, drift-diagnostic, and decision-policy foundations are complete.

| Phase | Status | Deliverable |
|---|---|---|
| Phase 1 | Complete | Local data workflow, EDA, feature engineering, and chronological model datasets |
| Phase 2 | Complete | Reproducible LightGBM baseline, fraud-focused evaluation, and versioned model bundle |
| Phase 3 | Complete | Probability calibration, drift diagnostics, policy optimization, and versioned policy bundle |
| Product integration | Planned | SHAP, API prediction, dashboard, monitoring, and deployment |

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
- Reloaded-bundle inference tests
- Automated pytest coverage

### Not Yet Implemented

- SHAP explanations and reason codes
- Production `/predict` integration
- Batch prediction integration
- Prediction logging and monitoring alerts
- Streamlit monitoring dashboard
- Threshold-simulation workflow
- Production cost-assumption validation
- Cloud deployment

The existing API and dashboard directories are scaffolds for later phases.

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

### Planned Product Layer

- FastAPI
- Pydantic
- SHAP
- Streamlit
- SQLite for local prediction logs
- PostgreSQL for deployment
- Docker

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

### 5. Train a local development model

Use `--skip-test-evaluation` during development to avoid repeatedly inspecting the frozen chronological test split:

```bash
python -m src.train \
  --model-output models/payguard_baseline_local.joblib \
  --model-version baseline-local-v1 \
  --n-jobs 4 \
  --skip-test-evaluation
```

The published `baseline-v1` test results are already recorded. Further implementation decisions should not be tuned against that test result.

### 6. Load a saved bundle

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

The baseline model bundle, calibrated policy bundle, and decision-engine foundations exist. API prediction integration, explanation, monitoring, and deployment remain future work.

---

## Planned API Capabilities

- Validate transaction payloads
- Load the versioned model bundle
- Produce fraud-risk scores
- Map scores to operational decisions
- Return explanation reason codes
- Log predictions and decisions
- Expose model metadata and health information

Planned endpoints include:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Service health |
| `GET` | `/model-info` | Model version and metadata |
| `POST` | `/predict` | Score one transaction |
| `POST` | `/batch-predict` | Score multiple transactions |
| `POST` | `/threshold-simulation` | Simulate decision-policy impact |
| `GET` | `/recent-predictions` | Supply monitoring data |

These endpoints are not all implemented yet.

---

## Project Structure

```text
api/          FastAPI application scaffold, schemas, and decision logic
dashboard/    Streamlit dashboard scaffold
data/         Local raw, processed, and sample-payload directories
db/           Prediction-log database components
docs/         Architecture, implementation plans, and model results
models/       Local serialized model artifacts
notebooks/    EDA and analytical notebooks
src/          Data processing, preprocessing, evaluation, training, and model bundles
tests/        Automated pytest suite
```

---

## Roadmap

- [x] Phase 1 — Data setup, EDA, feature engineering, and chronological datasets
- [x] Phase 2 — LightGBM baseline, evaluation, and versioned model bundle
- [x] Phase 3 — Probability calibration, drift analysis, and decision thresholds
- [ ] Phase 4 — SHAP explanations and reason codes
- [ ] Phase 5 — RiskFlow PayGuard API prediction integration
- [ ] Phase 6 — Monitoring and threshold-simulation dashboard
- [ ] Phase 7 — Docker and cloud deployment
- [ ] Phase 8 — Portfolio polish and product demonstration

---

## Current Limitations

The calibrated policy should not be treated as a deployed payment
authorization system.

Principal limitations include:

- Development cost assumptions are not validated against production data
- No final-test transaction reached the frozen block threshold
- No SHAP explanations or stable reason codes
- No dedicated high-value fraud objective
- No production API integration
- No live prediction logging or alerting
- No automated retraining workflow
- No formal policy approval or model-risk governance process

The recorded `baseline-v1` model and `calibrated-policy-v1` policy should
remain frozen benchmarks for future improvements.

---

## License

MIT
