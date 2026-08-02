
# Phase 3 — Calibration, Drift, and Decision Thresholds

> **Status: Planned**
>
> Phase 3 starts from the completed Phase 2 baseline merge commit
> `7f5ab99`.
>
> The frozen `baseline-v1` LightGBM model and
> `models/payguard_baseline.joblib` must not be modified or overwritten.
>
> The chronological test split was already evaluated once for the Phase 2
> baseline. It must not be used for Phase 3 development, calibrator
> selection, policy selection, or threshold tuning.

## Objective

Phase 3 converts the frozen Phase 2 ranking model into a versioned,
calibrated fraud-decision policy.

The phase will:

* partition the existing chronological validation period into isolated
  calibration-fit and policy-selection segments
* measure probability calibration
* compare deterministic calibration methods
* diagnose temporal score and feature drift
* define explicit operational and cost assumptions
* select deterministic `ALLOW`, `REVIEW`, and `BLOCK` thresholds
* persist a new calibrated policy bundle
* evaluate the frozen Phase 3 artifact once on the chronological test split
* document the resulting calibration, drift, and policy performance

Phase 3 does not retune the LightGBM model.

## Fixed Phase 2 baseline

Phase 3 uses the existing frozen baseline:

| Field                 |                             Value |
| --------------------- | --------------------------------: |
| Model type            |                          LightGBM |
| Model version         |                     `baseline-v1` |
| Feature count         |                                63 |
| Categorical features  |                                29 |
| Numerical features    |                                34 |
| Best iteration        |                             1,454 |
| Positive-class weight |                           27.4343 |
| Baseline artifact     | `models/payguard_baseline.joblib` |

The following components remain fixed:

* fitted LightGBM trees
* best iteration
* model hyperparameters
* categorical vocabularies
* missing-category handling
* unknown-category handling
* ordered feature contract
* training split
* Phase 2 validation and test results

The baseline bundle will only be loaded and used for inference.

## Phase boundaries

Phase 3 includes:

* deterministic validation partitioning
* probability-calibration metrics
* identity, sigmoid, and isotonic calibration
* deterministic calibration-method selection
* prediction-score drift analysis
* temporal performance analysis
* categorical missing and unknown-rate analysis
* explicit policy assumptions and constraints
* deterministic decision-threshold optimization
* calibrated policy persistence
* one-time final chronological test evaluation
* Phase 3 results and completion documentation

Phase 3 does not include:

* LightGBM hyperparameter tuning
* feature engineering changes
* model retraining
* repeated test-set evaluation
* SHAP explanations
* FastAPI `/predict` integration
* dashboard implementation
* online monitoring infrastructure
* production policy approval
* Kaggle leaderboard optimization

## Data-use policy

The existing chronological datasets remain:

```text
data/processed/train.parquet
data/processed/validation.parquet
data/processed/test.parquet
```

Their Phase 3 roles are:

| Dataset     | Phase 3 use                                                                         |
| ----------- | ----------------------------------------------------------------------------------- |
| Training    | No additional fitting; retained as model provenance                                 |
| Validation  | Calibration fitting, calibration selection, drift development, and policy selection |
| Test        | One final evaluation after the complete Phase 3 artifact is frozen                  |
| Kaggle test | Never used for internal development or evaluation                                   |

The Phase 2 model used the complete validation split for early stopping.
Consequently, Phase 3 validation results are development results rather than
a fully independent estimate of model ranking performance.

Splitting validation into two Phase 3 segments isolates calibrator fitting
from policy selection, but it does not undo the earlier use of validation for
Phase 2 early stopping.

## Validation development partition

The existing validation rows are already stored in chronological order.

Phase 3 will divide them by row position without shuffling:

1. `calibration_fit`

   * earliest `floor(validation_rows / 2)` rows
2. `policy_selection`

   * all remaining later validation rows

For the current validation dataset of 88,581 rows, the expected partition is:

| Segment          |   Rows |
| ---------------- | -----: |
| Calibration fit  | 44,290 |
| Policy selection | 44,291 |

Partition rules:

* preserve the materialized validation row order
* never shuffle rows
* never stratify by the target
* never move rows between the original train, validation, and test datasets
* preserve `TransactionID`
* require unique transaction IDs within and across both segments
* require both target classes in each segment
* require the complete Phase 2 feature contract
* return in-memory dataset objects rather than tracked derived Parquet files
* record partition metadata for reproducibility

The split position, row counts, transaction-ID boundaries, class counts, and
fraud rates must be available as structured metadata.

## Baseline scores

All Phase 3 scores originate from the frozen Phase 2 bundle.

For each development segment:

1. load `models/payguard_baseline.joblib`
2. validate the bundle and feature contract
3. apply the persisted Phase 2 categorical encoder
4. generate the positive-class LightGBM score
5. validate that every score is finite and within `[0, 1]`
6. preserve score order relative to the input transactions

The raw LightGBM score must remain available alongside any calibrated
probability.

A class-weighted LightGBM score must not be described as a literal fraud
probability before calibration has been evaluated.

## Calibration metrics

Calibration diagnostics will be reusable independently of LightGBM.

### Reliability table

The default reliability table will use ten fixed-width probability bins:

```text
[0.0, 0.1)
[0.1, 0.2)
...
[0.9, 1.0]
```

Each non-empty bin will report:

* lower probability bound
* upper probability bound
* transaction count
* transaction share
* mean predicted probability
* observed fraud rate
* absolute calibration gap
* fraudulent transaction count

Empty bins will remain identifiable but will not contribute to weighted
calibration metrics.

### Expected calibration error

Expected calibration error, or ECE, will be:

```text
sum(
    bin_count / total_count
    * abs(mean_predicted_probability - observed_fraud_rate)
)
```

Only non-empty bins contribute.

### Maximum calibration error

Maximum calibration error, or MCE, will be the largest absolute calibration
gap among non-empty bins.

### Calibration slope and intercept

Calibration slope and intercept will be estimated from:

```text
logit(observed outcome)
    ~ calibration_intercept
    + calibration_slope * logit(predicted probability)
```

Probabilities used by logarithmic calculations will be clipped to a documented
numerical epsilon.

Interpretation:

* ideal calibration intercept: `0`
* ideal calibration slope: `1`
* positive intercept: probabilities are generally too low
* negative intercept: probabilities are generally too high
* slope below `1`: predictions are too extreme
* slope above `1`: predictions are not sufficiently separated

The implementation must fail clearly for invalid inputs rather than silently
returning misleading metrics.

### Supporting metrics

Each calibration report will also include:

* fraud prevalence
* PR-AUC
* ROC-AUC
* log loss
* Brier score
* minimum probability
* maximum probability
* mean probability

PR-AUC and ROC-AUC remain ranking metrics. Calibration must not be accepted
solely because ranking metrics remain unchanged.

## Candidate calibrators

Phase 3 will compare three calibration methods.

### Identity

The identity method returns the frozen baseline score unchanged.

It provides:

* the no-calibration benchmark
* a fallback if fitted calibrators degrade development performance
* the lowest-complexity candidate

### Sigmoid

Sigmoid calibration will fit a one-dimensional logistic mapping using the
`calibration_fit` segment.

The mapping will be learned from the frozen baseline score and target only.
It must not fit or alter any LightGBM parameter.

### Isotonic

Isotonic calibration will fit a monotonic non-parametric mapping using the
`calibration_fit` segment.

Predictions outside the fitted score range will use explicit clipping
behaviour.

The implementation must preserve monotonic ordering and bound outputs within
`[0, 1]`.

## Calibration selection policy

Candidate calibrators will be fitted, where applicable, using only
`calibration_fit`.

All candidates will then be evaluated on `policy_selection`.

The selected method will minimize the following deterministic lexicographic
selection key:

1. policy-selection log loss
2. policy-selection Brier score
3. policy-selection ECE
4. candidate complexity

The complexity tie-break order is:

```text
identity
sigmoid
isotonic
```

Metric values used for deterministic comparisons will be rounded to a
documented precision before comparison.

Additional selection rules:

* non-finite probabilities disqualify a candidate
* probabilities outside `[0, 1]` disqualify a candidate
* candidate evaluation must use identical policy-selection rows
* PR-AUC and ROC-AUC will be recorded but will not drive calibration selection
* the chronological test split will not participate
* the selected calibrator will remain fitted only on `calibration_fit`
* the selected calibrator will not be refitted after inspecting
  `policy_selection`

The complete candidate comparison and selection reason must be recorded as
structured metadata.

## Temporal drift diagnostics

Phase 3 will add diagnostics for prediction drift, temporal performance drift,
and categorical preprocessing drift.

These diagnostics describe observed changes. They do not automatically prove
that the model is unsafe or identify the cause of a change.

### Raw-score PSI

Population stability index, or PSI, will compare frozen baseline-score
distributions.

The reference distribution will be `calibration_fit`.

Development comparison:

```text
calibration_fit -> policy_selection
```

Final comparison after artifact freeze:

```text
calibration_fit -> chronological test
```

Reference bin edges will be derived deterministically from reference-score
quantiles. Duplicate edges will be removed, and the outer bounds will cover
all possible score values.

The PSI calculation will use an explicit numerical epsilon for zero-count
bins.

### Calibrated-score PSI

The same PSI procedure will be applied to probabilities from the selected
calibrator.

Raw-score and calibrated-score PSI must be reported separately because a
calibration mapping can change the probability distribution without changing
the underlying model ranking.

### Temporal performance windows

Each evaluated segment will be divided into deterministic contiguous
row-count windows.

The default number of windows will be five, subject to validation that every
window has enough rows and both target classes for the requested metrics.

Each window will report:

* first and last transaction ID
* row count
* fraud count
* fraud prevalence
* mean raw score
* mean calibrated probability
* PR-AUC
* ROC-AUC
* log loss
* Brier score
* ECE
* decision counts and rates after policy selection

Windows must preserve chronological row order and must never be shuffled.

### Categorical missing and unknown rates

The persisted Phase 2 encoder will be used to identify:

* missing categorical values
* values mapped to the unknown-category code
* values mapped to known training categories

For every categorical feature and temporal segment, report:

* missing count and rate
* unknown count and rate
* known count and rate
* change from the calibration-fit reference rate

The diagnostics must distinguish missing values from genuinely unseen
categories.

No validation, policy-selection, or test category may be added to the frozen
training vocabulary.

## Decision-policy contract

Phase 3 will introduce a typed, validated policy contract.

### Threshold contract

The thresholds must satisfy:

```text
0 <= review_threshold <= block_threshold <= 1
```

Decision semantics remain:

```text
probability < review_threshold
    -> ALLOW

review_threshold <= probability < block_threshold
    -> REVIEW

probability >= block_threshold
    -> BLOCK
```

Thresholds apply to the selected calibrated probability, not directly to the
uncalibrated LightGBM score.

### Cost assumptions

Policy optimization must receive explicit cost assumptions rather than relying
on undocumented constants.

The initial contract will represent:

* cost of reviewing one transaction
* cost of blocking one legitimate transaction
* fraud loss when a fraudulent transaction is allowed
* expected fraud-capture rate for reviewed fraudulent transactions
* transaction-amount treatment used by the fraud-loss calculation

The realized development cost for each action will follow documented rules:

* legitimate `ALLOW`: no intervention cost
* fraudulent `ALLOW`: fraud-loss cost
* any `REVIEW`: review cost
* reviewed fraud not captured: residual fraud-loss cost
* legitimate `BLOCK`: false-positive block cost
* fraudulent `BLOCK`: prevented fraud loss

The assumptions are product-policy inputs, not learned model parameters.

The placeholder values currently present in application configuration must not
be treated as approved production assumptions without being copied into and
validated by the explicit Phase 3 policy contract.

### Operational constraints

The policy contract will support explicit constraints such as:

* maximum review rate
* maximum block rate
* maximum total intervention rate
* optional minimum review precision
* optional minimum block precision
* optional minimum fraud recall
* optional minimum fraudulent-amount capture

A candidate threshold pair that violates an active constraint is infeasible
regardless of its calculated cost.

All active and inactive constraints must be serialized with the selected
policy.

## Threshold search

Threshold optimization will use only:

* probabilities from the selected calibrator
* labels from `policy_selection`
* transaction amounts required by the policy contract
* explicit cost assumptions
* explicit operational constraints

The chronological test split will not be loaded by the development threshold
search.

### Candidate thresholds

Candidates will be generated deterministically from calibrated
policy-selection probabilities using a fixed quantile grid.

The candidate set will:

* include probability boundaries
* include documented fixed quantiles
* convert quantiles to observed probability values
* remove duplicate values
* sort values deterministically
* evaluate only pairs where
  `review_threshold <= block_threshold`

The grid size will be explicit and persisted in policy metadata.

### Optimization objective

Among feasible threshold pairs, select the pair with the lowest total policy
cost.

The report will also include:

* average cost per transaction
* allow count and rate
* review count and rate
* block count and rate
* confusion counts by decision
* fraud recall by intervention
* review precision
* block precision
* fraudulent amount captured
* comparison with an all-allow policy
* comparison with the current placeholder thresholds

### Deterministic tie-breaking

When candidate pairs have equal objective values at the documented comparison
precision, prefer:

1. fewer legitimate blocked transactions
2. lower total intervention rate
3. lower block rate
4. higher block threshold
5. higher review threshold

The complete search configuration, feasible-candidate count, selected
objective, and tie-break outcome must be recorded.

## Calibrated policy artifact

Phase 3 will create a new local artifact:

```text
models/payguard_calibrated_policy.joblib
```

Suggested version:

```text
calibrated-policy-v1
```

The artifact must not overwrite:

```text
models/payguard_baseline.joblib
```

The calibrated policy bundle will contain:

* a validated copy of or reference-compatible embedding of the frozen
  Phase 2 baseline bundle
* baseline model version
* calibrated policy version
* ordered raw feature contract
* selected calibrator name
* fitted calibrator state
* candidate calibration comparison
* calibration metric configuration
* review threshold
* block threshold
* policy cost assumptions
* operational constraints
* threshold-search configuration
* policy-selection metrics
* drift-reference metadata
* source dataset-manifest metadata
* creation timestamp
* bundle schema version

The bundle must provide deterministic inference from compatible raw features:

```text
raw features
    -> Phase 2 categorical encoding
    -> frozen LightGBM raw score
    -> calibrated fraud probability
    -> ALLOW / REVIEW / BLOCK decision
```

The inference output must expose both:

* raw model score
* calibrated fraud probability

Atomic persistence, overwrite protection, bundle validation, reload behaviour,
and post-load inference must be covered by automated tests.

Generated artifacts remain gitignored.

## Development and final-evaluation workflow

### Development workflow

Until the complete calibrator and policy are frozen, commands must use:

```text
--skip-test-evaluation
```

Development may load:

* the baseline artifact
* the Phase 1 manifest and feature metadata
* the chronological validation dataset

Development must not load or score the chronological test dataset.

### Freeze point

The Phase 3 artifact is frozen when:

* validation partitioning is finalized
* calibration metrics are finalized
* calibrator candidates are finalized
* calibrator selection is complete
* drift diagnostics are finalized
* policy assumptions are explicit
* operational constraints are explicit
* threshold-search behaviour is finalized
* all tests pass
* the calibrated policy artifact has been persisted and reloaded successfully
* the source commit is recorded
* no test metrics have influenced the artifact

### Final test evaluation

After the freeze point:

1. load the persisted calibrated policy artifact
2. load the chronological test split
3. score the test split once
4. calculate final calibration, ranking, drift, and policy metrics
5. write local machine-readable results
6. document the final results
7. do not change calibrator or threshold choices in response to the result

Any future policy or model iteration must use a new artifact version and a new
evaluation strategy rather than repeatedly tuning against this test result.

## Generated-output policy

Generated development outputs may include:

```text
models/payguard_calibrated_policy.joblib
reports/phase3_calibration.json
reports/phase3_policy.json
reports/phase3_test_evaluation.json
logs/phase3_*.log
```

These outputs must remain outside Git.

Tracked files should contain:

* source code
* tests
* human-written implementation documentation
* summarized final results
* reproducibility instructions

Tracked documentation must not depend on committing local IEEE-CIS data or
binary model artifacts.

## Implementation sequence

Each item will be completed as one isolated, validated commit.

### Commit 1 — Phase 3 plan

Add this documentation-only implementation plan.

No source code, tests, configuration, or generated artifacts change.

Suggested commit:

```text
docs: define phase 3 calibration and policy plan
```

### Commit 2 — Calibration development partition

Add deterministic chronological validation partitioning.

Tests will cover:

* odd and even row counts
* preserved row order
* expected segment sizes
* transaction-ID isolation
* both-class validation
* unchanged feature contract
* deterministic metadata

Suggested commit:

```text
feat: add calibration development partition
```

### Commit 3 — Calibration metrics

Add reliability tables, ECE, MCE, calibration slope, and calibration
intercept.

Tests will use small synthetic arrays with known expected results and cover
invalid probabilities, empty bins, and deterministic bin boundaries.

Suggested commit:

```text
feat: add probability calibration metrics
```

### Commit 4 — Fraud probability calibration

Add identity, sigmoid, and isotonic calibrators plus deterministic
development-only selection.

Tests will cover:

* fitted-state validation
* probability bounds
* monotonic isotonic outputs
* deterministic selection
* candidate tie-breaking
* serialization
* no test-data dependency

Suggested commit:

```text
feat: add fraud probability calibration
```

### Commit 5 — Temporal fraud drift diagnostics

Add raw-score PSI, calibrated-score PSI, chronological performance windows,
and categorical missing and unknown-rate diagnostics.

Tests will use small deterministic synthetic periods.

Suggested commit:

```text
feat: add temporal fraud drift diagnostics
```

### Commit 6 — Fraud decision policy optimization

Add typed cost assumptions, operational constraints, candidate generation,
policy metrics, and deterministic `ALLOW` / `REVIEW` / `BLOCK` threshold
search.

Tests will cover:

* threshold validation
* action-boundary semantics
* cost calculations
* infeasible candidates
* operational constraints
* deterministic tie-breaking
* all-allow comparison
* no test-data dependency

Suggested commit:

```text
feat: add fraud decision policy optimization
```

### Commit 7 — Calibrated policy persistence

Add the versioned calibrated policy bundle, atomic persistence, development
workflow, reload validation, and raw-feature inference.

The normal development command must default to skipping test evaluation or
require an explicit final-evaluation action.

Suggested commit:

```text
feat: persist calibrated fraud policy bundle
```

### Commit 8 — Phase 3 results

After the policy artifact is frozen, evaluate it once against the
chronological test split and document:

* selected calibration method
* calibration-fit metrics
* policy-selection metrics
* final test calibration metrics
* raw and calibrated score drift
* temporal-window performance
* categorical novelty diagnostics
* selected thresholds
* decision volumes
* expected policy cost
* fraud and fraudulent-amount capture
* comparison with the frozen Phase 2 baseline
* limitations

Suggested commit:

```text
docs: record calibration and policy results
```

### Commit 9 — Phase 3 completion

Update project-level roadmap and usage documentation after all Phase 3 code,
tests, artifact generation, and final results are complete.

Suggested commit:

```text
docs: complete phase 3 calibration and thresholds
```

## Validation requirements

Every Phase 3 implementation commit must pass:

```bash
python -m pytest
git diff --check
git status --short
```

Before committing, verify that no generated data, report, log, or model
artifact is staged.

Relevant ignore checks include:

```bash
git check-ignore data/processed/test.parquet
git check-ignore models/payguard_baseline.joblib
git check-ignore models/payguard_calibrated_policy.joblib
```

The standard automated test suite must:

* use synthetic fixtures
* remain deterministic
* avoid the full IEEE-CIS datasets
* avoid loading the chronological test split
* avoid writing tracked artifacts
* avoid modifying `baseline-v1`

## Phase 3 acceptance criteria

Phase 3 is complete when:

* the Phase 2 LightGBM model remains unchanged
* `baseline-v1` remains available and unmodified
* validation is partitioned chronologically and deterministically
* calibrator fitting and policy selection use separate validation segments
* calibration metrics are implemented and tested
* identity, sigmoid, and isotonic candidates are implemented
* calibration selection is deterministic
* raw and calibrated score drift are reported
* temporal performance windows are reported
* categorical missing and unknown rates are reported
* cost assumptions are explicit and validated
* operational constraints are explicit and validated
* threshold search is deterministic and development-only
* `ALLOW`, `REVIEW`, and `BLOCK` semantics remain compatible with the
  existing decision engine
* the calibrated policy artifact is versioned separately from the baseline
* the artifact can be reloaded and used for raw-feature inference
* the chronological test split is evaluated only after the artifact is frozen
* final test results do not trigger additional tuning
* the complete automated test suite passes
* generated datasets, reports, logs, and model artifacts remain outside Git
* final results and known limitations are documented
