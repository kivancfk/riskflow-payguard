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

```text
models/payguard_calibrated_policy.joblib

The artifact remained unchanged throughout Phase 7.

Deployment Topology

The local deployment topology is:

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

FastAPI and Streamlit are separate processes but use the same application
image.

Both connect directly to the shared PostgreSQL database.

The dashboard does not call the FastAPI service for monitoring data.

Application Image

The Dockerfile provides one immutable application image for both application
services.

The image:

uses python:3.11-slim
installs the libgomp1 runtime required by LightGBM
installs the pinned Python runtime dependencies
creates a non-root appuser
copies only application code and the frozen policy artifact
performs a build-time load_policy() integrity check
runs without source-code bind mounts
runs without a model-artifact bind mount
does not use the development Uvicorn --reload process

The default container command starts FastAPI.

Docker Compose overrides that command for the Streamlit service.

Frozen Artifact Protection

The image contains:

models/payguard_calibrated_policy.joblib

The policy loader verifies the expected SHA-256 before deserialization and
validates the frozen bundle metadata after loading.

The expected digest is:

5d53f23719ae891ecc24585393585765aa7fc0900ab38f95e37f59c18fe6c90f

A missing, replaced, empty, or incompatible artifact causes startup or image
build validation to fail rather than falling back to another policy.

The frozen thresholds and policy metadata are not exposed as runtime
configuration knobs.

Docker Build Context

.dockerignore uses an allowlist-style build context.

The image build receives only the files required by the deployed application:

requirements.txt
api/
dashboard/
src/
models/payguard_calibrated_policy.joblib

Local data, notebooks, tests, Git metadata, virtual environments, environment
files, and unrelated project files are not copied into the build context.

Runtime Configuration

Native local execution continues to default to:

DATABASE_URL=sqlite:///./predictions.db

Docker Compose supplies a PostgreSQL DATABASE_URL to the API and dashboard.

Local Compose bootstrap settings are documented in .env.example:

POSTGRES_DB
POSTGRES_USER
POSTGRES_PASSWORD

Production credentials must not be committed and should be supplied through a
deployment platform's secret/configuration mechanism.

Frozen model versions, calibration, thresholds, explanation versions,
reason-code versions, and artifact SHA-256 remain application-controlled
constants rather than environment configuration.

PostgreSQL Service

Docker Compose uses:

postgres:16

The PostgreSQL service:

is available only inside the Compose network
does not publish port 5432 to the host
uses a named volume for persisted database files
has a pg_isready health check
must become healthy before the application services start

Named volume:

payguard-postgres-data

Running:

docker compose down

preserves the database volume.

Running:

docker compose down --volumes

also removes the database volume.

FastAPI Service

The API service:

uses the shared application image
connects to PostgreSQL through DATABASE_URL
publishes port 8000
waits for PostgreSQL health
runs without --reload

Health endpoint:

GET /health

Model metadata endpoint:

GET /model-info

The API still loads and validates the frozen policy during application
lifespan startup.

Streamlit Service

The dashboard service:

uses the same application image as FastAPI
runs as a separate process
connects directly to the same PostgreSQL database
publishes port 8501
waits for PostgreSQL health
remains read-only with respect to policy configuration

Health endpoint:

/_stcore/health

The Monitoring and Threshold Simulator behavior remains unchanged from
Phase 6.

Candidate thresholds remain temporary and cannot be saved, applied, or
promoted to the frozen policy.

Persistence Portability

Native local development uses SQLite.

The containerized deployment path uses PostgreSQL.

The same SQLAlchemy prediction-event model supports both databases.

Phase 7 validated:

scoring-event insertion into PostgreSQL
frozen provenance persistence
string transaction identifiers
integer transaction identifiers
ground-truth label backfill
string-versus-integer transaction-ID distinction
persistence across container recreation

During deployment validation, mixed transaction-ID data exposed a
PostgreSQL-specific issue in the original typed lookup.

Casting all JSON transaction identifiers to integer could fail when unrelated
rows contained string identifiers.

The corrected lookup compares the serialized JSON scalar representation.

For example:

7001
"7001"

remain different identifiers without requiring PostgreSQL to cast arbitrary
stored strings to integers.

This persistence correction does not change inference behavior.

Health Checks

Docker Compose defines health checks for all three services.

PostgreSQL

Uses:

pg_isready
FastAPI

Requires:

http://localhost:8000/health

to return successfully from inside the container.

Streamlit

Requires:

http://localhost:8501/_stcore/health

to return successfully from inside the container.

Compose dependency ordering prevents API and dashboard startup from racing an
unready PostgreSQL service.

Deployment Smoke Validation

The deployment smoke harness is:

scripts/deployment_smoke.py

Run it from the repository root:

python scripts/deployment_smoke.py

The harness uses the isolated Compose project:

riskflow-payguard-smoke

This prevents the test from intentionally sharing the normal local Compose
project's PostgreSQL volume.

The smoke workflow:

validates the Compose configuration
removes stale resources belonging to the smoke project
builds the application image
starts PostgreSQL, FastAPI, and Streamlit
waits for all services to become healthy
validates FastAPI health
validates Streamlit health
validates the complete frozen /model-info contract
submits deterministic string and integer transaction IDs through /predict
verifies frozen prediction provenance
performs PostgreSQL ground-truth label backfill for both identifier types
verifies persisted rows directly through PostgreSQL
destroys and recreates containers without deleting the PostgreSQL volume
verifies the persisted rows remain available
removes the isolated containers, network, and PostgreSQL volume

A failure prints Compose service state and recent logs before cleanup.

Validated Smoke Result

The final Phase 7 smoke run completed with:

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

The smoke environment was fully removed afterward.

Automated Tests

The repository test suite remained green after Phase 7 deployment changes:

447 passed

The deployment smoke harness is intentionally separate from the normal pytest
suite.

Normal unit and integration tests therefore do not require Docker or a running
PostgreSQL instance.

Local Containerized Operation

Start the stack:

docker compose up \
  --build \
  --detach \
  --wait \
  --wait-timeout 120

Inspect services:

docker compose ps

FastAPI:

http://localhost:8000

Streamlit:

http://localhost:8501

Stop the stack while preserving PostgreSQL data:

docker compose down
Schema Management

The current stable prediction_events schema is initialized through
SQLAlchemy create_all.

No migration framework was added during Phase 7.

This is intentional for the first stable deployment schema.

Before the first post-deployment schema change, explicit migration tooling such
as Alembic should be introduced. create_all should not be treated as a schema
evolution mechanism for an established production database.

Cloud Deployment Boundary

Phase 7 prepares the application for a future managed deployment but does not
commit provider-specific infrastructure.

The current image is suitable for running FastAPI and Streamlit as separate
services backed by a managed PostgreSQL database.

A likely future topology is:

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

The frozen model artifact is intentionally excluded from Git.

A cloud build process therefore must either:

build from an environment where the verified frozen artifact is securely
available, or
deploy a prebuilt image from a trusted container registry

The repository does not add a provider manifest, Terraform, Kubernetes,
reverse proxy, Redis, Celery, or other infrastructure that is not currently
required.

Remaining Production Concerns

Phase 7 does not yet provide:

public cloud hosting
authentication or authorization
rate limiting
TLS or reverse-proxy configuration
automated monitoring alerts
automated ground-truth ingestion
backup and restore procedures
database migration tooling
production secret rotation
production traffic validation
model-risk or policy-approval governance
automated retraining

These remain separate future product and operational concerns.

Phase 7 Completion Criteria

Phase 7 is complete because:

the frozen inference contract remains unchanged
the application builds into a hardened container image
build-time policy integrity validation succeeds
FastAPI and Streamlit run as separate services from the same image
PostgreSQL is the validated containerized persistence backend
all Compose services have health checks
PostgreSQL data survives container recreation
typed transaction-ID label backfill works on PostgreSQL
the complete deployment smoke workflow passes
the normal automated test suite remains green
the frozen policy SHA-256 remains unchanged
