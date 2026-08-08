# Database

RiskFlow PayGuard persists successful scoring events for monitoring,
ground-truth labeling, and threshold simulation.

## Runtime Databases

### Native local development

The default database is SQLite:

```text
sqlite:///./predictions.db

The database file is created automatically when the application initializes
the persistence schema.

Containerized deployment

The Docker Compose deployment uses PostgreSQL 16.

FastAPI and Streamlit connect directly to the same PostgreSQL database through
DATABASE_URL.

The Compose stack constructs the application database URL from:

POSTGRES_DB
POSTGRES_USER
POSTGRES_PASSWORD

The PostgreSQL service is internal to the Compose network and is not published
to a host port.

Database files are stored in the named volume:

payguard-postgres-data

Running:

docker compose down

removes the containers and network but preserves the named database volume.

Running:

docker compose down --volumes

also deletes the persisted PostgreSQL volume.

Schema

The persisted table is:

prediction_events

It is defined declaratively in api/logging_db.py.

Each row represents one scoring event. Repeated scoring of the same
transaction creates another event rather than overwriting earlier history.

Column	Purpose
prediction_id	Autoincrementing primary key for the scoring event
transaction_id	JSON scalar preserving string-versus-integer identifier type
raw_model_score	Frozen baseline-model fraud score
calibrated_probability	Sigmoid-calibrated fraud probability
decision	Persisted ALLOW, REVIEW, or BLOCK decision
transaction_amount	Optional transaction amount used by monitoring/economic analysis
model_version	Frozen baseline model version
policy_version	Frozen policy version
calibration_method	Frozen calibration method
review_threshold	REVIEW threshold used for the event
block_threshold	BLOCK threshold used for the event
policy_artifact_sha256	SHA-256 of the frozen policy artifact
explanation_version	Explanation contract version
reason_code_version	Reason-code contract version
reason_codes	Persisted analyst reason-code list
scored_at_utc	UTC scoring timestamp
actual_label	Optional ground-truth label: 0 legitimate, 1 fraud
label_recorded_at_utc	UTC timestamp for the accepted ground-truth label

The table includes constraints for score ranges, threshold ordering,
transaction amount, decision values, and binary ground-truth labels.

Transaction-Identifier Portability

transaction_id accepts either a string or an integer.

Those types remain distinct:

42
"42"

are different transaction identifiers.

Ground-truth lookup compares the serialized JSON representation rather than
casting arbitrary stored identifiers to one scalar type. This keeps label
backfill portable across SQLite and PostgreSQL and avoids conflating string and
integer identifiers.

Persistence Semantics

Successful API prediction flow is:

validate
→ infer
→ persist
→ return HTTP response

For batch predictions, the batch persistence operation is atomic.

Persistence is fail-closed: a database persistence failure does not produce a
successful prediction response.

Ground-truth label batches are also atomic. Label handling rejects:

invalid non-binary labels
unknown transaction identifiers
conflicting relabels
non-UTC label timestamps

Replaying the same accepted label is idempotent.

Schema Creation and Migrations

The current stable schema is initialized with:

Base.metadata.create_all(...)

This is sufficient for the current first deployment schema.

No migration framework is configured yet.

Before the first post-deployment schema change, introduce an explicit migration
tool such as Alembic rather than relying on create_all to evolve an existing
database.

Deployment Validation

The PostgreSQL persistence path is exercised by:

python scripts/deployment_smoke.py

The smoke validation confirms:

PostgreSQL startup
API persistence
string and integer transaction-ID preservation
ground-truth label backfill
direct PostgreSQL row verification
persistence across container recreation
cleanup of the isolated smoke environment
