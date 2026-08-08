# Phase 7 — Deployment Foundation

## Status

Phase 7 is complete for the repository's containerized deployment scope.

This phase packages the existing RiskFlow PayGuard API and dashboard into a
repeatable Docker deployment and validates the persistence layer against
PostgreSQL.

Phase 7 does not retrain, recalibrate, retune, overwrite, or promote any model
or policy artifact.

It also does not claim that the application is already running as a public
cloud production service.

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

~~~text
models/payguard_calibrated_policy.joblib
~~~

The artifact remained unchanged throughout Phase 7.

## Deployment Topology

The local deployment topology is:

~~~text
                         Docker Compose
                              |
          +-------------------+-------------------+
          |                   |                   |
          v                   v                   v
     PostgreSQL 16         FastAPI            Streamlit
       postgres              api              dashboard
          ^                   |                   |
          |                   |                   |
          +-------------------+-------------------+
                    shared DATABASE_URL
~~~

FastAPI and Streamlit are separate processes but use the same application
image.

Both connect directly to the shared PostgreSQL database.

The dashboard does not call the FastAPI service for monitoring data.

## Application Image

The `Dockerfile` provides one immutable application image for both application
services.

The image:

- uses `python:3.11-slim`
- installs the `libgomp1` runtime required by LightGBM
- installs the pinned Python runtime dependencies
- creates a non-root `appuser`
- copies only application code and the frozen policy artifact
- performs a build-time `load_policy()` integrity check
- runs without source-code bind mounts
- runs without a model-artifact bind mount
- does not use the development Uvicorn `--reload` process

The default container command starts FastAPI.

Docker Compose overrides that command for the Streamlit service.

## Frozen Artifact Protection

The image contains:

~~~text
models/payguard_calibrated_policy.joblib
~~~

The policy loader verifies the expected SHA-256 before deserialization and
validates the frozen bundle metadata after loading.

Expected digest:

~~~text
5d53f23719ae891ecc24585393585765aa7fc0900ab38f95e37f59c18fe6c90f
~~~

A missing, replaced, empty, or incompatible artifact causes startup or image
build validation to fail rather than falling back to another policy.

The frozen thresholds and policy metadata are not exposed as runtime
configuration knobs.

## Docker Build Context

`.dockerignore` uses an allowlist-style build context.

The image build receives only the files required by the deployed application:

~~~text
requirements.txt
api/
dashboard/
src/
models/payguard_calibrated_policy.joblib
~~~

Local data, notebooks, tests, Git metadata, virtual environments, environment
files, and unrelated project files are not copied into the build context.

## Runtime Configuration

Native local execution continues to default to:

~~~text
DATABASE_URL=sqlite:///./predictions.db
~~~

Docker Compose supplies a PostgreSQL `DATABASE_URL` to the API and dashboard.

Local Compose bootstrap settings are documented in `.env.example`:

~~~text
POSTGRES_DB
POSTGRES_USER
POSTGRES_PASSWORD
~~~

Production credentials must not be committed and should be supplied through a
deployment platform's secret/configuration mechanism.

Frozen model versions, calibration, thresholds, explanation versions,
reason-code versions, and artifact SHA-256 remain application-controlled
constants rather than environment configuration.

## PostgreSQL Service

Docker Compose uses:

~~~text
postgres:16
~~~

The PostgreSQL service:

- is available only inside the Compose network
- does not publish port 5432 to the host
- uses a named volume for persisted database files
- has a `pg_isready` health check
- must become healthy before the application services start

Named volume:

~~~text
payguard-postgres-data
~~~

Running:

~~~bash
docker compose down
~~~

preserves the database volume.

Running:

~~~bash
docker compose down --volumes
~~~

also removes the database volume.

## FastAPI Service

The API service:

- uses the shared application image
- connects to PostgreSQL through `DATABASE_URL`
- publishes port `8000`
- waits for PostgreSQL health
- runs without `--reload`

Health endpoint:

~~~text
GET /health
~~~

Model metadata endpoint:

~~~text
GET /model-info
~~~

The API still loads and validates the frozen policy during application
lifespan startup.

## Streamlit Service

The dashboard service:

- uses the same application image as FastAPI
- runs as a separate process
- connects directly to the same PostgreSQL database
- publishes port `8501`
- waits for PostgreSQL health
- remains read-only with respect to policy configuration

Health endpoint:

~~~text
/_stcore/health
~~~

The Monitoring and Threshold Simulator behavior remains unchanged from
Phase 6.

Candidate thresholds remain temporary and cannot be saved, applied, or
promoted to the frozen policy.

## Persistence Portability

Native local development uses SQLite.

The containerized deployment path uses PostgreSQL.

The same SQLAlchemy prediction-event model supports both databases.

Phase 7 validated:

- scoring-event insertion into PostgreSQL
- frozen provenance persistence
- string transaction identifiers
- integer transaction identifiers
- ground-truth label backfill
- string-versus-integer transaction-ID distinction
- persistence across container recreation

During deployment validation, mixed transaction-ID data exposed a
PostgreSQL-specific issue in the original typed lookup.

Casting JSON transaction identifiers to integer could fail when unrelated rows
contained string identifiers.

The corrected lookup compares the serialized JSON scalar representation.

For example:

~~~text
7001
"7001"
~~~

remain different identifiers without requiring PostgreSQL to cast arbitrary
stored strings to integers.

This persistence correction does not change inference behavior.

## Health Checks

Docker Compose defines health checks for all three services.

### PostgreSQL

Uses `pg_isready`.

### FastAPI

Requires the internal endpoint:

~~~text
http://localhost:8000/health
~~~

to return successfully.

### Streamlit

Requires:

~~~text
http://localhost:8501/_stcore/health
~~~

to return successfully.

Compose dependency ordering prevents API and dashboard startup from racing an
unready PostgreSQL service.

## Deployment Smoke Validation

The deployment smoke harness is:

~~~text
scripts/deployment_smoke.py
~~~

Run it from the repository root:

~~~bash
python scripts/deployment_smoke.py
~~~

The harness uses the isolated Compose project:

~~~text
riskflow-payguard-smoke
~~~

This prevents the smoke test from intentionally sharing the normal local
Compose project's PostgreSQL volume.

The workflow:

1. validates the Compose configuration
2. removes stale smoke-project resources
3. builds the application image
4. starts PostgreSQL, FastAPI, and Streamlit
5. waits for all services to become healthy
6. validates FastAPI health
7. validates Streamlit health
8. validates the frozen `/model-info` contract
9. submits deterministic string and integer transaction IDs through `/predict`
10. verifies frozen prediction provenance
11. performs PostgreSQL label backfill for both identifier types
12. verifies persisted rows directly through PostgreSQL
13. recreates containers without deleting the PostgreSQL volume
14. verifies that persisted rows remain available
15. removes the isolated containers, network, and PostgreSQL volume

A failure prints Compose service state and recent logs before cleanup.

## Validated Smoke Result

The final Phase 7 smoke run completed with:

~~~text
health checks: ok
frozen model-info: ok
PostgreSQL prediction persistence: ok
typed PostgreSQL label backfill: ok
direct PostgreSQL verification: ok
recreating containers without deleting PostgreSQL volume...
health checks: ok
frozen model-info: ok
direct PostgreSQL verification: ok
deployment smoke: PASS
~~~

The smoke environment was fully removed afterward.

## Automated Tests

The repository test suite remained green after Phase 7 deployment changes:

~~~text
447 passed
~~~

The deployment smoke harness is intentionally separate from the normal pytest
suite.

Normal tests therefore do not require Docker or a running PostgreSQL instance.

## Local Containerized Operation

Start the stack:

~~~bash
docker compose up \
  --build \
  --detach \
  --wait \
  --wait-timeout 120
~~~

Inspect services:

~~~bash
docker compose ps
~~~

FastAPI:

~~~text
http://localhost:8000
~~~

Streamlit:

~~~text
http://localhost:8501
~~~

Stop the stack while preserving PostgreSQL data:

~~~bash
docker compose down
~~~

## Schema Management

The current stable `prediction_events` schema is initialized through SQLAlchemy
`create_all`.

No migration framework was added during Phase 7.

This is intentional for the first stable deployment schema.

Before the first post-deployment schema change, explicit migration tooling such
as Alembic should be introduced.

`create_all` should not be treated as a schema-evolution mechanism for an
established production database.

## Cloud Deployment Boundary

Phase 7 prepares the application for a future managed deployment but does not
commit provider-specific infrastructure.

A future topology can use the same image for separate FastAPI and Streamlit
services backed by managed PostgreSQL:

~~~text
container registry
      |
      +----> FastAPI service
      |
      +----> Streamlit service

managed PostgreSQL
      ^
      |
      +---- FastAPI
      |
      +---- Streamlit
~~~

The frozen model artifact is intentionally excluded from Git.

A cloud build therefore must either:

- build where the verified frozen artifact is securely available, or
- deploy a trusted prebuilt image from a container registry

The repository does not add provider-specific infrastructure that is not yet
required, including Kubernetes, Terraform, a reverse proxy, Redis, or Celery.

## Remaining Production Concerns

Phase 7 does not yet provide:

- public cloud hosting
- authentication or authorization
- rate limiting
- TLS or reverse-proxy configuration
- automated monitoring alerts
- automated ground-truth ingestion
- backup and restore procedures
- database migration tooling
- production secret rotation
- production traffic validation
- model-risk or policy-approval governance
- automated retraining

These remain separate future product and operational concerns.

## Phase 7 Completion Criteria

Phase 7 is complete when:

- the frozen inference contract remains unchanged
- the application builds into the hardened container image
- build-time policy integrity validation succeeds
- FastAPI and Streamlit run as separate services from the same image
- PostgreSQL is the validated containerized persistence backend
- all Compose services have health checks
- PostgreSQL data survives container recreation
- typed transaction-ID label backfill works on PostgreSQL
- the deployment smoke workflow passes
- the normal automated test suite remains green
- the frozen policy SHA-256 remains unchanged

All of these criteria were satisfied during Phase 7 validation.
