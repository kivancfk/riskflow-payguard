# Database

The predictions database backs the dashboard's monitoring views and stores ground-truth labels for offline model evaluation.

- **Local development:** SQLite (`predictions.db` in the project root, created automatically on first API run).
- **Deployment:** PostgreSQL. Set `DATABASE_URL` in the environment, e.g.

  ```text
  DATABASE_URL=postgresql+psycopg2://user:pass@host:5432/riskflow
  ```

## Schema

The `predictions` table is defined declaratively in `api/logging_db.py`. SQLAlchemy creates it on startup via `Base.metadata.create_all(engine)`. Columns:

| Column | Type | Notes |
|---|---|---|
| `transaction_id` | string, PK | Merchant-side id, idempotency key |
| `fraud_probability` | float | Calibrated probability in [0, 1] |
| `decision` | string | `ALLOW` / `REVIEW` / `BLOCK` |
| `model_version` | string | Version string from the model artifact |
| `features` | JSON | Input payload |
| `top_risk_factors` | JSON | SHAP-derived top-k features |
| `scored_at` | datetime | UTC timestamp |
| `actual_label` | float, nullable | Backfilled later for offline evaluation |

## Migrations

The scaffold uses `create_all` for simplicity. If the schema starts changing in non-trivial ways, swap in Alembic — add an `alembic/` folder here and an `alembic.ini` at the project root.
