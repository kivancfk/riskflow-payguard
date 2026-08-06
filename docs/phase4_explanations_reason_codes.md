# Phase 4 — SHAP Explanations and Reason Codes

> **Status: Complete**
>
> Phase 4 starts from the completed Phase 3 merge commit
> `5ec1670`.
>
> The frozen `baseline-v1` LightGBM model and
> `calibrated-policy-v1` decision policy must not be retrained, recalibrated,
> retuned, overwritten, or otherwise modified during this phase.
>
> SHAP values explain signals used by the frozen LightGBM model. They do not
> prove that a transaction is fraudulent and must not be presented as causal
> evidence.

## Objective

Phase 4 adds deterministic model explanations to the frozen fraud-scoring
workflow:

```text
transaction
    -> frozen LightGBM score
    -> calibrated fraud probability
    -> ALLOW / REVIEW / BLOCK decision
    -> SHAP contributions and analyst-facing reason codes
```

The phase will:

extract LightGBM SHAP contributions
identify the strongest score-increasing and score-decreasing signals
map model features to stable analyst-facing reason codes
distinguish observed, missing, and unknown-category inputs
guarantee deterministic contribution and reason ordering
reconstruct and validate the frozen LightGBM score
integrate explanations with calibrated policy predictions
verify that model and policy artifacts remain immutable
document the final explanation behaviour and limitations

Phase 4 does not change model predictions or decision thresholds.

Frozen model and policy

Phase 4 uses the following frozen inputs:

Field	Value
Model type	LightGBM binary classifier
Model version	baseline-v1
Baseline artifact	models/payguard_baseline.joblib
Policy version	calibrated-policy-v1
Calibration method	sigmoid
Review threshold	0.16255069862369795
Block threshold	0.8509223095305902
Policy artifact	models/payguard_calibrated_policy.joblib
Policy artifact SHA-256	5d53f23719ae891ecc24585393585765aa7fc0900ab38f95e37f59c18fe6c90f
Explanation method	LightGBM TreeSHAP
Explanation version	shap-explanation-v1
Reason-code version	reason-codes-v1

The following components remain fixed:

fitted LightGBM trees
model objective and best iteration
model hyperparameters
ordered feature contract
categorical training vocabularies
missing-category encoding
unknown-category encoding
sigmoid calibrator parameters
review and block thresholds
policy costs and constraints
model and policy version identifiers
persisted model and policy artifact bytes

Explanation code may load these components for inference but must never persist
changes back into either artifact.

Interpretation contract

The explanation contract distinguishes three score representations.

Raw model margin

The LightGBM raw margin is the model output before the binary objective link
function is applied.

For a binary LightGBM model, TreeSHAP contributions reconstruct this margin:

raw_model_margin
    = expected_value_raw
    + sum(feature_shap_values_raw)

Positive SHAP values increase the raw model margin relative to the expected
value. Negative SHAP values decrease it.

Raw model score

The existing Phase 2 and Phase 3 raw model score is the uncalibrated
LightGBM positive-class score.

It must be consistent with the reconstructed margin through the frozen
LightGBM objective transformation:

raw_model_score
    = lightgbm_link(raw_model_margin)

For the current binary objective, this transformation is the logistic sigmoid.

Calibrated fraud probability

The Phase 3 sigmoid calibrator transforms the frozen raw model score:

calibrated_probability
    = phase3_calibrator(raw_model_score)

SHAP values do not directly decompose the calibrated probability.

The calibrator changes the probability scale while preserving the model-score
ranking. Reason codes therefore describe influential frozen-model signals,
not additive changes in calibrated probability.

Explanation output contract

An explanation result will accompany one frozen policy prediction.

The result will contain:

transaction_id
model_version
policy_version
explanation_version
reason_code_version
raw_model_margin
raw_model_score
calibrated_probability
decision
expected_value_raw
shap_sum_raw
reconstructed_raw_margin
reconstructed_raw_model_score
margin_reconstruction_error
score_reconstruction_error
top_positive_contributions
top_negative_contributions
reason_codes

The existing policy fields must have the same values as a prediction generated
without explanations.

Contribution record

Each selected contribution will contain:

feature
feature_index
feature_group
direction
shap_value_raw
absolute_shap_value_raw
value_state
rank

Allowed direction values are:

INCREASES_SCORE
DECREASES_SCORE

Allowed value_state values are:

OBSERVED
MISSING
UNKNOWN_CATEGORY

Raw feature values will not be embedded in analyst-facing reason messages by
default. This prevents encoded category sentinels, device identifiers, email
domains, or other potentially sensitive values from being mistaken for
human-readable conclusions.

Contribution selection

The default explanation will return:

up to three positive contributions
up to three negative contributions

The expected-value column returned by LightGBM contribution prediction is not
a feature contribution and must be excluded from feature ranking.

A contribution is eligible only when:

its feature belongs to the frozen ordered feature contract
its value is finite
its absolute value is greater than 1e-12
its direction matches the requested positive or negative list

The implementation must not replace missing contributions with fabricated
reasons.

Deterministic contribution ordering

Positive contributions will be sorted by:

larger SHAP value
lower frozen feature index
lexical feature name

Equivalent sort key:

(-shap_value_raw, feature_index, feature)

Negative contributions will be sorted by:

more negative SHAP value
lower frozen feature index
lexical feature name

Equivalent sort key:

(shap_value_raw, feature_index, feature)

The returned rank is one-based within its positive or negative list.

Repeated explanations of the same input, model, and policy must produce the
same values, ordering, reason codes, and messages.

Stable feature groups

Reason codes will use stable feature groups rather than exposing every model
feature as a separate public API contract.

The initial mappings are:

Feature pattern	Stable feature group
TransactionAmt	TRANSACTION_AMOUNT
TransactionDT	TRANSACTION_TIME
ProductCD, card1–card6	PAYMENT_INSTRUMENT
addr1, addr2	ADDRESS
dist1, dist2	DISTANCE
P_emaildomain, R_emaildomain	EMAIL_DOMAIN
C1–C14	COUNT_AGGREGATE
D1–D15	TIME_DELTA
M1–M9	MATCH_FLAG
V1–V339 features retained by the model	ANONYMIZED_BEHAVIOR
DeviceType, DeviceInfo, id_*	DEVICE_IDENTITY
Any explicitly supported feature not covered above	OTHER_MODEL_SIGNAL

Mapping precedence will be:

exact feature-name mapping
documented prefix or numeric-range mapping
OTHER_MODEL_SIGNAL fallback

The fallback prevents a compatible frozen model feature from disappearing from
an explanation merely because it lacks a specialized analyst label.

Feature-group mappings must be versioned and reviewed. A label or mapping
change requires a new reason-code version.

Reason-code grammar

Reason codes will use the following deterministic grammar:

<FEATURE_GROUP>_<VALUE_STATE>_<DIRECTION>

Examples:

TRANSACTION_AMOUNT_OBSERVED_INCREASES_SCORE
DEVICE_IDENTITY_UNKNOWN_CATEGORY_INCREASES_SCORE
EMAIL_DOMAIN_MISSING_DECREASES_SCORE
ANONYMIZED_BEHAVIOR_OBSERVED_INCREASES_SCORE

The code describes:

the broad family of the influential model feature
whether the original input was observed, missing, or unknown
whether the feature increased or decreased the frozen model score

It must not state that the feature caused fraud.

Analyst-facing messages

Messages will be generated from stable templates rather than free-form text.

Example templates are:

Transaction amount signal increased the model score.
Device identity contained an unknown category and increased the model score.
Email-domain information was missing and decreased the model score.
An anonymized behavioral signal increased the model score.

Messages must use language such as:

increased the model score
decreased the model score
was influential in the model output
contained a missing value
contained a category not seen in training

Messages must not use unsupported statements such as:

this transaction is fraudulent because
this feature caused fraud
this proves fraud
the customer intentionally
the transaction should be blocked because of this feature alone
Missing-value handling

Missing-value state must be determined from the original feature input before
categorical encoding.

Values recognized as missing include the missing representations already
supported by the frozen feature and encoder contracts.

Rules:

missing values remain valid model inputs when allowed by the frozen contract
the frozen LightGBM missing-value path remains unchanged
the SHAP contribution is retained
the contribution receives value_state=MISSING
the reason code includes the MISSING state
missing values are not converted into unknown categories for explanation
purposes
no imputation may be introduced solely for SHAP calculation

A missing feature must not be confused with an entirely absent required
feature. Missing required columns remain a schema error.

Unknown-category handling

Unknown-category state must be detected using the frozen Phase 2 training
vocabularies before or while applying the persisted encoder.

Rules:

no new category may be added to a frozen vocabulary
the existing unknown-category code remains unchanged
the encoded sentinel must not be presented as the original feature value
the SHAP contribution is retained
the contribution receives value_state=UNKNOWN_CATEGORY
the reason code includes the UNKNOWN_CATEGORY state
unknown categories remain distinct from missing values

A category is unknown only when it is present in the input but absent from the
frozen training vocabulary.

Reason selection and ordering

Reason codes will be derived from ranked feature contributions.

The default result will include reasons corresponding to the selected positive
and negative contribution lists.

When multiple selected features produce the same reason code:

keep the feature with the largest absolute SHAP contribution
break equal-contribution ties using the lower frozen feature index
emit the reason code only once

Final reason ordering will be:

score-increasing reasons
score-decreasing reasons
larger absolute SHAP contribution
lower frozen feature index
lexical reason code

Equivalent conceptual sort key:

(
    direction_priority,
    -absolute_shap_value_raw,
    feature_index,
    reason_code,
)

where INCREASES_SCORE has priority over DECREASES_SCORE.

Reason ordering must not depend on dictionary ordering, set iteration,
DataFrame column accidents, thread scheduling, or platform-specific hash
behaviour.

Raw-score reconstruction checks

Every explanation must validate the TreeSHAP additive identity.

The implementation will compare:

expected_value_raw + sum(feature_shap_values_raw)

with the frozen LightGBM raw-margin prediction.

It will also transform the reconstructed margin through the frozen LightGBM
objective link and compare it with the existing raw model score.

Default numerical tolerances will be explicitly defined in code and tested.
The initial target tolerance is:

absolute error <= 1e-8

The explanation must fail clearly when:

contribution dimensions do not match the feature contract
a contribution or expected value is non-finite
the reconstructed margin exceeds tolerance
the reconstructed raw model score exceeds tolerance
a batch explanation changes input row order
the LightGBM output shape is unsupported

Reconstruction failures must not silently return approximate reason codes.

Artifact immutability

Phase 4 explanation generation is read-only with respect to model artifacts.

Before explanation development and integration, tests will record:

policy artifact SHA-256
baseline artifact SHA-256
model version
policy version
calibrator method
review threshold
block threshold
frozen feature ordering

After explanation generation, the same properties must remain unchanged.

The calibrated policy artifact must continue to match:

5d53f23719ae891ecc24585393585765aa7fc0900ab38f95e37f59c18fe6c90f

Explanation code must not:

call model fitting methods
call calibrator fitting methods
alter categorical vocabularies
alter model parameters
alter thresholds
overwrite either artifact
persist a modified copy under an existing frozen artifact path
fit a SHAP background distribution from validation or test data
use labels to generate an individual prediction explanation
Policy integration contract

The explanation-enabled prediction path will follow this order:

validate raw features
    -> apply frozen categorical encoding
    -> generate frozen raw model score
    -> generate calibrated probability
    -> apply frozen decision thresholds
    -> calculate SHAP contributions
    -> validate reconstruction
    -> generate deterministic reason codes

The explanation step is observational. It must not feed values back into score,
calibration, or decision calculations.

For identical input rows, the score-only and explanation-enabled paths must
produce identical:

raw model scores
calibrated probabilities
decisions
input row ordering

Explanation failures must be explicit. They must not replace a model decision
with a different decision or silently return an unvalidated explanation.

Batch behaviour

Batch explanations must:

preserve input row order
preserve transaction identifiers
produce one explanation per input row
use the same frozen feature ordering for every row
validate the complete contribution matrix shape
isolate reason ranking within each transaction
avoid state leakage between transactions
produce the same result whether a row is explained alone or in a batch
Phase boundaries

Phase 4 includes:

an explanation data contract
LightGBM TreeSHAP extraction
raw-margin and raw-score reconstruction
top positive and negative contribution selection
stable feature-group mapping
versioned reason-code generation
missing-value state handling
unknown-category state handling
deterministic ordering and deduplication
model and policy immutability checks
calibrated-policy integration
focused and integration tests
Phase 4 results and completion documentation

Phase 4 does not include:

model retraining
feature engineering changes
hyperparameter tuning
calibrator refitting
threshold revision
cost-assumption revision
repeated chronological test evaluation for model selection
causal fraud explanations
counterfactual explanations
model-independent surrogate models
analyst feedback collection
FastAPI response-schema changes beyond the explanation contract
Streamlit dashboard implementation
online explanation monitoring
production policy approval
Planned implementation structure

The expected implementation may introduce:

src/explanations.py
src/reason_codes.py
tests/test_explanations.py
tests/test_reason_codes.py
tests/test_explanation_integration.py
docs/phase4_results.md

Exact modules may be adjusted if the existing source structure provides a
clearer ownership boundary. Public contracts must remain typed, focused, and
independently testable.

Validation strategy

Phase 4 will be developed through isolated commits.

Focused validation will cover:

explanation result and contribution contracts
LightGBM contribution extraction
raw-margin reconstruction
raw model-score reconstruction
positive and negative contribution ranking
stable feature-group mappings
missing-value reason generation
unknown-category reason generation
deterministic reason ordering and deduplication
batch-order preservation
score-only versus explanation-path parity
model and policy artifact immutability
persisted policy reload and explanation integration

The full test suite will run only after focused tests for each change pass.

Completion criteria

Phase 4 is complete when:

the explanation contract is implemented and documented
SHAP values reconstruct the frozen LightGBM raw margin within tolerance
the reconstructed margin reproduces the frozen raw model score
positive and negative contributions are deterministic
reason-code mappings are versioned and stable
missing and unknown-category states are represented correctly
reason messages avoid causal claims
score-only and explanation-enabled predictions are identical
the frozen artifact hashes and policy values remain unchanged
focused explanation tests pass
the complete repository test suite passes
final Phase 4 results are documented

## Phase 4 completion record

Phase 4 was completed on 6 August 2026.

Implemented deliverables:

- immutable explanation and contribution contracts
- native LightGBM TreeSHAP extraction
- deterministic positive and negative contribution ranking
- observed, missing, and unknown-category value states
- stable feature-group mappings
- versioned analyst reason codes and messages
- reason-code deduplication
- raw-margin reconstruction
- raw model-score reconstruction
- calibrated-policy explanation integration
- score-only and explanation-enabled output parity
- deterministic batch and individual-row behaviour
- policy artifact reload parity
- frozen artifact and model-state immutability checks

Frozen configuration:

- model version: `baseline-v1`
- policy version: `calibrated-policy-v1`
- explanation version: `shap-explanation-v1`
- reason-code version: `reason-codes-v1`
- reconstruction tolerance: `1e-8`
- review threshold: `0.16255069862369795`
- block threshold: `0.8509223095305902`
- policy SHA-256:
  `5d53f23719ae891ecc24585393585765aa7fc0900ab38f95e37f59c18fe6c90f`

Focused Phase 4 validation contains 147 tests. The complete repository
test suite also passed before Phase 4 completion.

See [`phase4_results.md`](phase4_results.md) for the implementation
summary, validation evidence, interpretation guidance, and limitations.
