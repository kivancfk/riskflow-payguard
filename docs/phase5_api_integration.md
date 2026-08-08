# Phase 5 — FastAPI Integration

> **Status: Complete — 2026-08-08**
>
> Phase 5 started from the completed Phase 4 merge commit
> `789210f` and was completed on branch
> `phase-5-api-integration`.
>
> The frozen `baseline-v1` LightGBM model,
> `calibrated-policy-v1` decision policy, explanation contract, and
> reason-code contract must not be retrained, recalibrated, retuned,
> overwritten, or otherwise modified during this phase.

## Objective

Phase 5 exposes the completed frozen fraud-scoring pipeline through a
production-oriented FastAPI application.

The target request path is:

```text
HTTP request
    -> strict Pydantic validation
    -> ordered frozen-model feature frame
    -> frozen calibrated policy inference
    -> deterministic explanation generation
    -> typed JSON response

The underlying inference path remains:

raw model features
    -> frozen categorical encoder
    -> frozen LightGBM raw score
    -> sigmoid-calibrated probability
    -> ALLOW / REVIEW / BLOCK decision
    -> native LightGBM TreeSHAP contributions
    -> reconstruction validation
    -> deterministic analyst-facing reason codes

Phase 5 is an interface and integration phase. It does not alter the
statistical model, calibration, thresholds, explanation mappings, or policy
economics established in earlier phases.

Frozen inference assets

Phase 5 uses the following immutable inputs:

Field	Frozen value
Baseline model version	baseline-v1
Policy version	calibrated-policy-v1
Calibration method	sigmoid
Review threshold	0.16255069862369795
Block threshold	0.8509223095305902
Explanation version	shap-explanation-v1
Reason-code version	reason-codes-v1
Policy artifact	models/payguard_calibrated_policy.joblib
Policy SHA-256	5d53f23719ae891ecc24585393585765aa7fc0900ab38f95e37f59c18fe6c90f

The API may load and inspect the frozen artifact but must never write it back
to disk.

The following components are immutable during Phase 5:

fitted LightGBM trees
baseline model version
best iteration
frozen categorical encoder
categorical vocabularies
ordered feature contract
missing-category encoding
unknown-category encoding
sigmoid calibrator and its fitted parameters
review threshold
block threshold
policy costs and constraints
explanation version
reason-code version
reason-code mappings and messages
persisted model and policy artifact bytes
Existing integration boundary

Phase 5 must reuse the completed core inference interfaces rather than
reimplementing their behavior inside the API.

The frozen policy artifact is loaded through:

load_calibrated_policy_bundle(...)

The primary prediction and explanation interface is:

predict_policy_with_explanations(...)

That function already returns aligned:

transaction identifiers
raw model scores
calibrated probabilities
policy decisions
model and policy versions
explanation and reason-code versions
raw model margins
TreeSHAP contribution rankings
reconstruction diagnostics
analyst-facing reasons

The API layer is responsible for validation, feature-frame construction,
lifecycle management, serialization, and HTTP behavior. It is not responsible
for reproducing model, calibrator, threshold, SHAP, or reason-code logic.

Existing API scaffold

The repository already contains an earlier api/ scaffold.

Relevant files include:

api/
    __init__.py
    config.py
    decision_engine.py
    logging_db.py
    main.py
    model_loader.py
    schemas.py

The scaffold predates the completed frozen Phase 2–4 inference pipeline and
must therefore be treated as code to refactor rather than as the authoritative
scoring implementation.

In particular, the Phase 5 application must not use:

independently configured review or block thresholds
api.decision_engine.decide() for production decisions
the old generic model artifact loader
placeholder feature generation
prediction database logging
threshold simulation
recent-prediction database endpoints

Policy decisions must come directly from the frozen
CalibratedPolicyBundle.

Legacy modules may remain in the repository when deleting them would create
unnecessary unrelated churn, but they must not be part of the Phase 5
application request path.

Application architecture

The Phase 5 application is divided into four responsibilities.

1. Frozen policy loading

FastAPI lifespan startup will load the frozen calibrated policy exactly once
per application process.

Before inference is accepted, startup must:

resolve the configured frozen policy path;
require a non-empty regular file;
calculate its SHA-256 digest;
require the digest to equal the frozen Phase 5 digest;
load it with load_calibrated_policy_bundle(...);
validate its baseline model version;
validate its policy version;
validate its selected calibration method;
validate its review threshold;
validate its block threshold.

A missing, unreadable, corrupted, replaced, or incompatible artifact is a
startup failure.

The service must not silently fall back to another model or policy.

The loader must not call any training, calibration, policy-search, or artifact
save function.

2. Request validation

API requests will use Pydantic v2 models with extra fields forbidden.

A transaction request contains:

transaction_id
features

transaction_id is request metadata and must not become a model feature.

It follows the Phase 4 identifier contract:

string or integer
booleans are invalid
strings must be non-empty
strings must not contain surrounding whitespace

features contains the complete frozen model feature contract.

All frozen feature keys are required in the HTTP request.

This produces an intentional distinction:

field absent
    -> invalid API request

field present with null
    -> explicit missing feature value

Numerical and categorical fields may allow explicit missing values when the
frozen model contract supports them.

Malformed values, unsupported value types, booleans masquerading as numerical
values, and non-finite numerical inputs must be rejected before inference.

Unknown categorical values are not validation errors merely because they were
not observed during training. They are passed to the frozen encoder, which
maps them to its existing unknown-category representation.

Missing categorical values are passed to the same encoder and use its
existing missing-category representation.

The API must never extend or modify a categorical vocabulary.

3. Ordered feature-frame construction

Validated feature payloads will be converted to a pandas DataFrame.

The final model-frame column order must come from:

bundle.baseline_bundle.feature_columns

rather than from an independently maintained API ordering.

For every request:

set of validated API feature keys
    == set of frozen model feature names

and the generated frame must be reordered exactly to the frozen model feature
sequence before inference.

A single request produces one row.

A batch request produces rows in request order.

No Phase 5 code may derive new model features, remove model features, rename
features, reorder the frozen contract, or change model-facing values for
feature-engineering purposes.

4. Response serialization

API response models translate the immutable Phase 4 output objects into JSON.

Serialization must not recompute scores, decisions, contributions, or reason
codes.

The single-prediction response contains:

transaction_id
model_version
policy_version
explanation_version
reason_code_version
raw_model_score
calibrated_probability
decision
top_positive_contributions
top_negative_contributions
reason_codes
reasons
reconstruction

decision is one of:

ALLOW
REVIEW
BLOCK

Each contribution exposes the existing Phase 4 contribution contract:

feature
feature_index
feature_group
direction
shap_value_raw
absolute_shap_value_raw
value_state
rank

Allowed direction values remain:

INCREASES_SCORE
DECREASES_SCORE

Allowed value states remain:

OBSERVED
MISSING
UNKNOWN_CATEGORY

Analyst reasons expose stable reason codes and deterministic non-causal
messages produced by the existing Phase 4 mapping.

reason_codes preserves reason-code order as generated by the explanation
pipeline.

The reconstruction object exposes:

raw_model_margin
expected_value_raw
shap_sum_raw
reconstructed_raw_margin
reconstructed_raw_model_score
margin_reconstruction_error
score_reconstruction_error

These are diagnostic values describing reconstruction of the frozen raw
LightGBM model output. They do not decompose the calibrated probability.

The API will not add a generated scoring timestamp to prediction responses in
Phase 5. A request-time timestamp is unrelated to model inference and would
make otherwise identical repeated responses differ.

Endpoint contract

Phase 5 exposes four application endpoints.

GET /health

Purpose:

confirm the API process is running;
confirm the frozen policy was successfully loaded.

The endpoint does not run model inference.

A successful application startup implies that artifact integrity and frozen
policy compatibility checks have already passed.

GET /model-info

Returns immutable information describing the loaded inference policy.

The response includes, at minimum:

baseline_model_version
policy_version
calibration_method
review_threshold
block_threshold
explanation_version
reason_code_version
policy_artifact_sha256
feature_names
feature_count
categorical_feature_count
numerical_feature_count

Thresholds and version values must be derived from or verified against the
loaded frozen artifact. They must not come from an independent mutable
decision configuration.

POST /predict

Accepts one transaction request.

Processing is:

validate payload
    -> build one-row ordered feature frame
    -> call predict_policy_with_explanations(...)
    -> serialize the single aligned explanation record

The returned transaction identifier must exactly match the validated request
identifier.

POST /batch-predict

Accepts:

transactions: [...]

The batch must contain at least one transaction.

Transaction identifiers must satisfy the Phase 4 batch identifier contract,
including uniqueness within the batch.

The output must contain the same number of predictions as inputs.

Output position i must correspond to input position i.

The endpoint will make one aligned batch inference call rather than
implementing a separate scoring algorithm.

For every transaction, the result must equal the result obtained by scoring
the same transaction through the single-request path, excluding only
batch-wrapper structure.

Validation and HTTP behavior

Pydantic request-validation failures return FastAPI's standard 422
validation response.

The API must reject, before scoring:

absent required feature fields
extra feature fields
malformed transaction identifiers
unsupported field types
booleans supplied as numerical values
non-finite numerical values
empty batches
duplicate transaction identifiers within one batch

Explicit missing values that are valid under the frozen feature contract are
not equivalent to absent fields.

Unknown categorical values that can be represented by the frozen encoder are
not rejected merely for being unseen during training.

Unexpected failures after validated input reaches the trusted inference layer
must not be converted into successful responses.

Determinism contract

For the same:

frozen policy artifact
API code version
transaction identifier
feature values
input ordering

repeated requests must return the same model-facing prediction and
explanation content.

This includes:

raw model score
calibrated probability
decision
contribution values
contribution ordering
reason-code ordering
reason messages
reconstruction metadata
model version
policy version
explanation version
reason-code version

The application must not introduce nondeterministic explanation ordering,
new category mappings, random sampling, mutable threshold configuration, or
request-time model changes.

Parity requirements
Direct policy parity

For the same ordered feature frame, API results must match direct
predict_policy_with_explanations(...) inference.

Parity covers:

raw model score
calibrated probability
decision
transaction identifier
version fields
positive contributions
negative contributions
reasons
reconstruction metadata
Single versus batch parity

Scoring a transaction through /predict must produce the same prediction and
explanation content as scoring that transaction as part of
/batch-predict.

Order preservation

Batch requests and responses must preserve row order exactly.

No sorting by score, probability, decision, identifier, or explanation rank is
allowed at the transaction level.

Artifact immutability

The frozen policy artifact SHA-256 before Phase 5 API tests is:

5d53f23719ae891ecc24585393585765aa7fc0900ab38f95e37f59c18fe6c90f

API startup and prediction tests must verify that the artifact remains
byte-for-byte unchanged.

The Phase 5 code path must never call:

save_model_bundle(...)
save_calibrated_policy_bundle(...)

or any equivalent persistence path for the frozen artifact.

Test strategy

Phase 5 tests use FastAPI's TestClient for application-level behavior and
focused unit tests where lifecycle or serialization logic is easier to isolate.

Coverage will include:

Startup and loading
correct frozen artifact loads successfully
artifact is loaded once per application lifecycle
wrong SHA-256 is rejected
missing artifact is rejected
invalid artifact is rejected
wrong model version is rejected
wrong policy version is rejected
wrong calibration method is rejected
changed thresholds are rejected
Informational endpoints
/health succeeds after valid startup
/model-info exposes the expected frozen metadata
model information matches the loaded artifact
no scoring or artifact mutation occurs
Request validation
complete valid transaction accepted
absent feature rejected
extra feature rejected
malformed identifier rejected
incorrectly typed field rejected
invalid numerical value rejected
explicit missing value handled according to contract
unknown categorical value reaches the frozen encoder correctly
Single prediction
transaction identifier preserved
raw score matches direct inference
calibrated probability matches direct inference
decision matches direct inference
explanation versions match
positive contributions match
negative contributions match
reasons match
reconstruction metadata matches
repeated identical requests are deterministic
Batch prediction
response count equals input count
transaction identifiers preserve input order
model outputs preserve input order
explanations preserve input order
batch values match direct batch inference
each batch row matches equivalent single inference
duplicate transaction identifiers are rejected
Immutability and regressions
policy artifact SHA-256 unchanged after startup
policy artifact SHA-256 unchanged after requests
existing Phase 2–4 inference tests continue to pass
full repository test suite passes
Phase boundaries

Phase 5 does not include:

model retraining
feature-engineering changes
categorical-vocabulary updates
calibrator refitting
review-threshold revision
block-threshold revision
policy-cost changes
policy-constraint changes
reason-code mapping changes
explanation-version changes
database prediction logging
prediction-history APIs
monitoring dashboards
threshold simulation
repeated chronological test-set evaluation
Docker changes
cloud deployment

Existing repository files related to later concerns must not be expanded as
part of this phase.

Implementation sequence

Phase 5 was implemented as isolated commits.

Commit 1 — planning and API contract
docs: define phase 5 API contract

Define the frozen assets, endpoint contract, request and response behavior,
architecture, parity requirements, test strategy, and phase boundaries.

No endpoint implementation is included.

Commit 2 — frozen policy loader
feat: load and validate frozen API policy

Refactor API configuration and loading around
models/payguard_calibrated_policy.joblib.

Add:

SHA-256 verification
frozen metadata checks
one-time in-process policy storage
loader-focused tests

No prediction endpoint is implemented.

Commit 3 — strict schemas and feature frames
feat: define strict API schemas and feature frames

Add:

complete typed transaction feature schema
forbidden extra fields
strict identifier validation
explicit missing-value behavior
batch request schema
contribution/reason/reconstruction response schemas
ordered feature-frame construction
schema and frame unit tests

No scoring endpoint is implemented.

Commit 4 — application lifecycle and information endpoints
feat: expose API health and model information

Refactor FastAPI startup to load the verified frozen policy exactly once.

Implement:

GET /health
GET /model-info

Remove legacy database and independent-threshold behavior from the active
application path.

Add focused endpoint tests.

Commit 5 — single prediction endpoint
feat: integrate single transaction prediction

Implement:

POST /predict

Use predict_policy_with_explanations(...) as the sole inference integration
boundary.

Add direct score, decision, explanation, reconstruction, identifier, and
determinism parity tests.

Commit 6 — batch prediction endpoint
feat: integrate ordered batch prediction

Implement:

POST /batch-predict

Add:

input/output order preservation
identifier preservation
duplicate identifier validation
direct batch parity
single-versus-batch parity tests
Commit 7 — API hardening and immutability regression
test: harden API parity and artifact immutability

Complete cross-cutting tests for:

malformed requests
missing values
unknown categories
repeated requests
artifact SHA-256 immutability
frozen policy metadata
broader Phase 2–4 regressions
Commit 8 — Phase 5 completion documentation
docs: complete phase 5 API integration

Record final endpoint behavior, test results, artifact integrity, limitations,
and handoff state for the next phase.


## Completion record

Phase 5 is complete.

The FastAPI application now exposes exactly four Phase 5 business-facing
endpoints:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Confirm successful startup and frozen-policy loading |
| `GET` | `/model-info` | Expose frozen policy and feature-contract metadata |
| `POST` | `/predict` | Score and explain one validated transaction |
| `POST` | `/batch-predict` | Score and explain an ordered transaction batch |

The application startup path is fail-closed. FastAPI lifespan startup resolves
the configured policy path, calculates and validates the frozen SHA-256 digest
before deserialization, loads the calibrated policy once per application
process, validates the frozen model and policy metadata, and validates the API
feature contract before accepting requests.

The active application path does not use the legacy independent decision
engine, prediction database logging, threshold-simulation endpoint, or
recent-predictions endpoint.

### Frozen artifact verification

The completed API uses the following frozen contract:

| Field | Frozen value |
|---|---|
| Baseline model version | `baseline-v1` |
| Policy version | `calibrated-policy-v1` |
| Calibration method | `sigmoid` |
| Review threshold | `0.16255069862369795` |
| Block threshold | `0.8509223095305902` |
| Explanation version | `shap-explanation-v1` |
| Reason-code version | `reason-codes-v1` |
| Policy artifact | `models/payguard_calibrated_policy.joblib` |
| Policy SHA-256 | `5d53f23719ae891ecc24585393585765aa7fc0900ab38f95e37f59c18fe6c90f` |

The artifact digest was verified repeatedly before and after API startup,
single prediction, batch prediction, parity testing, hardening testing, and the
final repository regression run.

No Phase 5 application request or test changed the persisted artifact bytes.

### Request contract implemented

The API exposes the complete frozen 63-feature model contract:

- 63 total model features
- 29 categorical features
- 34 numerical features
- every model feature key required in HTTP requests
- explicit `null` values distinguished from absent fields
- extra fields forbidden
- malformed transaction identifiers rejected
- booleans rejected as numerical inputs
- non-finite numerical values rejected
- unseen categorical values delegated to the frozen encoder
- missing categorical values delegated to the frozen encoder
- frozen categorical vocabularies never expanded during inference

Model-facing DataFrame columns are ordered from
`bundle.baseline_bundle.feature_columns`, leaving the frozen model bundle as
the authoritative feature-order source.

### Prediction contract implemented

Both prediction endpoints reuse `predict_policy_with_explanations(...)` as the
sole model, calibrated-policy, decision, and explanation integration boundary.

The API does not independently calculate:

- raw model scores
- calibrated probabilities
- policy decisions
- SHAP contributions
- reconstruction diagnostics
- analyst reason codes

Single and batch responses expose:

- transaction identifier
- model version
- policy version
- explanation version
- reason-code version
- raw model score
- calibrated probability
- `ALLOW`, `REVIEW`, or `BLOCK` decision
- ordered positive TreeSHAP contributions
- ordered negative TreeSHAP contributions
- ordered reason codes
- deterministic analyst reason messages
- raw-margin and raw-score reconstruction metadata

Prediction responses intentionally contain no request-time scoring timestamp,
so repeated identical requests produce deterministic model-facing JSON.

### Batch behavior implemented

`POST /batch-predict`:

- requires at least one transaction
- rejects duplicate transaction identifiers
- performs one aligned batch inference call
- preserves request row order
- preserves string and integer identifier types
- returns exactly one output row for each input row
- matches equivalent `/predict` results at the row level

### Validation and HTTP behavior

Malformed request payloads are rejected through FastAPI/Pydantic validation
with HTTP `422` before the trusted inference boundary runs.

This includes incomplete feature sets, extra fields, invalid identifiers,
incorrect field types, invalid numerical values, empty batches, and duplicate
batch identifiers.

Explicit missing values remain valid when permitted by the frozen feature
contract.

Unseen categorical values are not rejected solely because they were absent
during training; the existing frozen encoder maps them to its unknown-category
representation.

### Determinism and parity verification

Phase 5 tests verify:

- API versus direct `predict_policy_with_explanations(...)` parity
- single versus batch prediction parity
- transaction identifier preservation
- batch row-order preservation
- repeated-request determinism
- deterministic contribution ordering
- deterministic reason-code ordering
- reconstruction invariants
- encoder-vocabulary immutability
- persisted policy-artifact immutability
- exactly one Phase 4 inference-boundary call per prediction request
- malformed requests do not reach model inference

### Final regression evidence

The final Phase 5 repository regression run completed successfully:

`383 passed in 32.34s`

The final frozen policy SHA-256 remained:

`5d53f23719ae891ecc24585393585765aa7fc0900ab38f95e37f59c18fe6c90f`

The final pre-documentation implementation head was:

`49dca80 test: harden API parity and artifact immutability`

`git diff --check` passed and the working tree was clean before this
documentation update.

### Phase 5 implementation commits

Phase 5 was implemented through the following sequence:

- `9ea3a7d` — `docs: define phase 5 API contract`
- `708a0e1` — `feat: load and validate frozen API policy`
- `e9323b8` — `feat: define strict API schemas and feature frames`
- `51c0442` — `feat: expose API health and model information`
- `bd915bd` — `feat: integrate single transaction prediction`
- `947ad63` — `feat: integrate ordered batch prediction`
- `bafa629` — `test: fix API prediction test whitespace`
- `49dca80` — `test: harden API parity and artifact immutability`

The small `bafa629` follow-up removed one extra EOF blank line detected by
`git diff --check`. It made no functional change.

### Handoff

Phase 5 leaves a deterministic local FastAPI inference surface over the frozen
Phase 2–4 model, calibrated policy, and explanation assets.

Later phases may add prediction persistence, monitoring, dashboarding,
threshold simulation, Docker packaging, authentication, deployment, and
production operational controls.

Those later concerns must continue to treat `baseline-v1`,
`calibrated-policy-v1`, the frozen thresholds, explanation mappings, and the
persisted artifact bytes as immutable benchmarks unless a future explicitly
versioned model or policy replaces them.

## Completion criteria

Phase 5 is complete only when:

FastAPI loads the correct frozen policy safely at startup;
artifact SHA-256 verification succeeds;
the policy artifact remains unchanged;
/health works;
/model-info reports the frozen inference contract;
/predict returns typed policy and explanation results;
/batch-predict preserves identifiers and input order;
request validation rejects incomplete, extra, malformed, and incorrectly
typed payloads;
missing and unknown categorical values follow the frozen encoder contract;
API raw scores match direct frozen-policy inference;
API calibrated probabilities match direct frozen-policy inference;
API decisions match direct frozen-policy inference;
API explanations match direct explanation inference;
repeated requests are deterministic;
batch predictions match equivalent single predictions;
focused Phase 5 tests pass;
existing Phase 2–4 regression tests pass;
git diff --check passes;
no out-of-scope Phase 5 functionality is introduced.
