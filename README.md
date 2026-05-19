# RiskFlow PayGuard
## Real-Time Payment Fraud Risk Scoring System

RiskFlow PayGuard is a real-time fraud risk scoring system for payment transactions. It provides a FastAPI service that scores incoming transactions, returns an `ALLOW`, `REVIEW`, or `BLOCK` decision, explains the top contributing risk factors, logs every prediction, and exposes a Streamlit dashboard for fraud monitoring, threshold tuning, and business-impact analysis.

The goal is to build a deployed, demoable fintech-style product rather than only a notebook-based machine learning project.

### Why This Project Matters

In modern payments, fraud decisions cannot stop at a probability score. Every transaction must be turned into an operational action — allow, review, or block — and every action carries a cost: missed fraud, blocked legitimate customers, or unnecessary manual review work. RiskFlow PayGuard demonstrates the full path from raw transaction data to a calibrated risk score, a business decision, an explainable response, and a monitoring dashboard, mirroring how real payment risk products operate.

---

## Stack

- **API:** FastAPI + Pydantic
- **Modeling:** LightGBM
- **Explainability:** SHAP
- **Dashboard:** Streamlit
- **Database:** SQLite for local development, PostgreSQL for deployment
- **Deployment:** Docker, Render, Streamlit Community Cloud
- **Data:** IEEE-CIS Fraud Detection dataset from Kaggle

> **Note:** Raw Kaggle data is not committed to this repository. See `data/README.md` for download instructions.

---

## Components

### RiskFlow PayGuard API

FastAPI service that scores payment transactions in real time.

Main responsibilities:

- Validate transaction payloads
- Generate fraud probability
- Map probability to risk band and business decision
- Return top contributing risk factors
- Log every prediction

### RiskFlow PayGuard Dashboard

Streamlit dashboard for fraud monitoring and threshold tuning.

Main responsibilities:

- Monitor prediction volume and risk distribution
- Analyze allow / review / block decisions
- Simulate fraud-catching performance under different thresholds
- Estimate false-positive cost and manual-review workload

---

## Live Demo

- **API:** coming soon
- **Dashboard:** coming soon

---

## Architecture

```text
client
  ↓
RiskFlow PayGuard API  ──  /predict
  ↓
model + SHAP
  ↓
decision engine
  ↓
response

prediction logs
  ↓
predictions database  ──  SQLite (local) / PostgreSQL (deployed)
  ↓
RiskFlow PayGuard Dashboard
```

---

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Liveness check |
| `GET` | `/model-info` | Model version, training date, and metrics |
| `POST` | `/predict` | Score a single transaction |
| `POST` | `/batch-predict` | Score a batch of transactions |
| `POST` | `/threshold-simulation` | Simulate fraud caught, false-positive cost, and review workload |
| `GET` | `/recent-predictions` | Recent predictions (for the dashboard) |

---

## Local Setup

**1. Install dependencies:**

```bash
pip install -r requirements.txt
```

**2. Run the API:**

```bash
uvicorn api.main:app --reload
```

The API will be available at: `http://localhost:8000`
Interactive docs at: `http://localhost:8000/docs`

**3. Run the dashboard:**

```bash
streamlit run dashboard/app.py
```

The dashboard will be available at: `http://localhost:8501`

> **Database note:** Local development uses SQLite by default (created automatically on first run). For deployment, set the `DATABASE_URL` environment variable to a PostgreSQL connection string.

---

## Project Structure

```text
api/         RiskFlow PayGuard API — FastAPI service, schemas, decision logic, prediction logging
dashboard/   RiskFlow PayGuard Dashboard — Streamlit monitoring and threshold simulation app
notebooks/   EDA, feature engineering, training, threshold simulation, and explainability notebooks
src/         Reusable Python modules for features, training, calibration, SHAP, and utilities
models/      Serialized model artifacts
data/        raw/, processed/, and sample_payloads/
db/          Database schema and initialization
tests/       pytest suite
docs/        Architecture and analysis documentation
```

---

## Roadmap

- [ ] Phase 1 — Data setup and EDA
- [ ] Phase 2 — Feature engineering
- [ ] Phase 3 — Modeling with LightGBM and probability calibration
- [ ] Phase 4 — Explainability with SHAP
- [ ] Phase 5 — RiskFlow PayGuard API
- [ ] Phase 6 — RiskFlow PayGuard Dashboard
- [ ] Phase 7 — Deployment
- [ ] Phase 8 — Portfolio polish

---

## License

MIT
