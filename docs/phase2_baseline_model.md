# Phase 2 — Baseline Model

## Objective

Phase 2 produces the first reproducible LightGBM fraud classifier for RiskFlow PayGuard.

The baseline must:

* train from the materialized Phase 1 datasets
* preserve the chronological train, validation, and test split
* use the existing 63-feature contract
* handle numerical missing values and categorical features consistently
* address class imbalance without changing the chronological data distribution
* produce fraud-focused evaluation metrics
* save a versioned model bundle that can later be loaded by the API
* remain deterministic enough for local development and automated testing

Phase 2 is intentionally a baseline-model phase rather than a hyperparameter-optimization phase.

## Phase boundaries

Phase 1 already completed the initial feature engineering and materialized the model-ready datasets.

Phase 2 includes:

* processed-dataset loading and schema validation
* categorical preprocessing
* LightGBM baseline training
* early stopping on the validation split
* validation and final test evaluation
* model artifact serialization
* model artifact loading tests
* baseline results documentation

Phase 2 does not include:

* probability calibration
* SHAP explanations
* production API prediction integration
* business decision-threshold optimization
* extensive hyperparameter tuning
* Kaggle leaderboard optimization

Those concerns will be handled in later phases.

## Input datasets

The training workflow will consume:

```text
data/processed/train.parquet
data/processed/validation.parquet
data/processed/test.parquet
data/processed/feature_metadata.csv
data/processed/dataset_manifest.json
```

The unlabeled file below will not be used for model selection or internal evaluation:

```text
data/processed/kaggle_test.parquet
```

Expected labeled dataset structure:

```text
TransactionID
isFraud
63 model features
```

Feature groups:

* 29 categorical features
* 34 numerical features

## Data-split policy

The chronological split created in Phase 1 is fixed:

* training: 413,378 rows
* validation: 88,581 rows
* test: 88,581 rows

Usage rules:

1. The training split is used to fit preprocessing state and model parameters.
2. The validation split is used for early stopping and baseline design decisions.
3. The test split remains untouched until the implementation and parameters are frozen.
4. The Kaggle test set is never used for internal model evaluation.
5. Rows must never be randomly redistributed between the existing splits.

This prevents future information from leaking into earlier training periods.

## Feature contract

The training workflow must read the expected feature names and feature types from the Phase 1 metadata rather than maintaining a second independent feature list.

Before training, it must validate:

* required files exist
* target exists only where expected
* identifier and target columns are not model inputs
* feature names match the metadata
* feature order is deterministic
* no duplicate feature names exist
* target contains only binary values
* datasets contain at least one positive and one negative observation
* train, validation, and test schemas are compatible

A schema mismatch must fail loudly before LightGBM training begins.

## Numerical-feature handling

Numerical features will:

* be converted to a consistent numerical dtype
* preserve missing values for LightGBM's native missing-value handling
* avoid mean or median imputation in the first baseline
* avoid scaling because tree-based models do not require standardized feature ranges

Infinite values must be detected and converted to missing values or rejected through an explicit validation rule.

## Categorical-feature handling

Categorical preprocessing state must be learned from the training split only.

The workflow will:

1. normalize missing categorical values to a dedicated missing token
2. create deterministic training-category vocabularies
3. map categories to integer codes
4. reserve a code for categories not observed during training
5. apply the same mappings to validation, test, and future inference rows
6. persist the mappings inside the model artifact

Validation or test categories must not expand the training vocabulary.

This preprocessing contract is required so that offline training and future API inference use identical category mappings.

## Class-imbalance strategy

The first baseline will use class weighting calculated from the training target:

```text
negative training rows / positive training rows
```

The resulting value will be supplied to LightGBM through its positive-class weighting configuration.

The baseline will not use SMOTE or random oversampling because:

* the data is chronological
* synthetic transactions may distort the fraud distribution
* class weighting is simpler to reproduce in production
* the original probability distribution should remain available for later calibration analysis

The calculated class weight must be recorded in the model metadata.

## Baseline LightGBM design

The model will use `LGBMClassifier` with:

* binary classification objective
* conservative tree complexity
* learning rate below the LightGBM default
* a sufficiently high maximum estimator count
* validation-based early stopping
* explicit random seeds
* controlled thread configuration
* class weighting derived from the training split
* categorical feature indicators supplied explicitly

The initial parameters will be manually selected as a defensible baseline.

Hyperparameter-search frameworks will not be added during this phase.

## Evaluation policy

Accuracy is not an appropriate primary metric because fraud represents approximately 3.5% of transactions.

### Primary metric

```text
PR-AUC / average precision
```

PR-AUC reflects the model's ability to rank rare fraudulent transactions ahead of legitimate transactions.

### Supporting probability and ranking metrics

The evaluation report will include:

* ROC-AUC
* PR-AUC
* binary log loss
* Brier score
* positive-class prevalence

### Reference threshold metrics

At a reference probability threshold of `0.50`, report:

* confusion matrix
* precision
* recall
* F1 score
* false-positive rate
* false-negative rate

The `0.50` threshold is diagnostic only. It is not the final PayGuard allow, review, or block policy.

### Fraud-operations metrics

The baseline should also report fraud recall at fixed review-capacity levels, for example:

* top 0.5% of transactions
* top 1% of transactions
* top 2% of transactions
* top 5% of transactions

Where the transaction-amount field is available, the report should additionally calculate fraudulent amount captured at those review-capacity levels.

These metrics connect model performance to a practical manual-review queue.

## Validation and test workflow

Training will proceed in two stages.

### Stage 1 — Development evaluation

* fit on the training split
* use the validation split for early stopping
* report validation metrics
* inspect feature importance and basic prediction sanity checks
* freeze preprocessing and model parameters

### Stage 2 — Final baseline evaluation

After the implementation is frozen:

* evaluate the selected model once on the chronological test split
* record test metrics separately from validation metrics
* do not make additional model choices based on repeated test inspection

Retraining on combined train and validation data will be considered only after the baseline test result has been recorded.

## Artifact contract

The training command will create a local, gitignored artifact such as:

```text
models/payguard_baseline.joblib
```

The serialized bundle will contain:

* fitted LightGBM classifier
* ordered feature names
* categorical feature names
* numerical feature names
* categorical vocabularies
* missing and unknown category codes
* model parameters
* class weight
* best iteration
* validation metrics
* test metrics
* training timestamp
* model version
* source dataset-manifest metadata

The artifact must be loadable without access to the training DataFrames.

Loading the artifact and scoring a small compatible DataFrame must be covered by an automated test.

## Command-line interface

The intended training entry point is:

```bash
python -m src.train
```

Useful arguments should include:

```text
--processed-dir
--model-output
--seed
--n-jobs
--overwrite
--skip-test-evaluation
```

Safety rules:

* refuse to overwrite an existing model unless `--overwrite` is supplied
* provide clear errors for missing Phase 1 artifacts
* print the active dataset and model configuration
* print validation metrics before any test evaluation
* print the saved artifact path after successful completion

## Implementation sequence

Each item should be completed as a separate, validated commit.

### Commit 1 — Phase 2 plan

Add this implementation plan without changing training code.

Suggested commit:

```text
docs: define phase 2 baseline model plan
```

### Commit 2 — Dataset contract and loader

Add reusable processed-data loading and validation functions.

Tests should cover:

* missing files
* incorrect target values
* feature mismatch
* incorrect feature ordering
* train, validation, and test schema compatibility

Suggested commit:

```text
feat: add baseline dataset loader
```

### Commit 3 — Categorical preprocessing

Add training-only category vocabularies and deterministic category encoding.

Tests should cover:

* missing values
* unseen categories
* deterministic mappings
* unchanged feature order
* serialization compatibility

Suggested commit:

```text
feat: add categorical feature encoding
```

### Commit 4 — Fraud evaluation metrics

Add reusable threshold-free, threshold-based, and review-capacity metrics.

Tests should use small synthetic prediction arrays with known expected results.

Suggested commit:

```text
feat: add fraud model evaluation metrics
```

### Commit 5 — LightGBM training pipeline

Replace the placeholder training entry point with the deterministic baseline pipeline.

Tests should train only on small synthetic datasets. The full IEEE-CIS dataset must not be required by the standard automated test suite.

Suggested commit:

```text
feat: implement LightGBM baseline training
```

### Commit 6 — Artifact persistence and loading

Save the complete inference bundle and verify that it can be reloaded and used for prediction.

Suggested commit:

```text
feat: persist baseline model bundle
```

### Commit 7 — Full baseline run and documentation

Run the model against the local Phase 1 datasets and document:

* model parameters
* best iteration
* training duration
* validation metrics
* test metrics
* review-capacity results
* known limitations

Generated Parquet and model binary files must remain outside Git.

Suggested commit:

```text
docs: record baseline model results
```

### Commit 8 — Phase completion documentation

Update the README roadmap and usage instructions only after the baseline implementation and tests are complete.

Suggested commit:

```text
docs: complete phase 2 baseline model
```

## Validation requirements

Every implementation commit must pass:

```bash
python -m pytest
```

Additional checks should include:

```bash
git diff --check
git status --short
```

Before committing, verify that no generated data or model artifact is staged:

```bash
git status --short
git check-ignore data/processed/train.parquet
git check-ignore models/payguard_baseline.joblib
```

## Phase 2 acceptance criteria

Phase 2 is complete when:

* processed-data schemas are validated before training
* categorical preprocessing is learned from training data only
* the baseline LightGBM model trains reproducibly
* early stopping uses the validation split
* PR-AUC and supporting metrics are reported
* review-capacity metrics are reported
* the chronological test split is evaluated only after the implementation is frozen
* the complete model and preprocessing state are serialized together
* the artifact can be reloaded and used for prediction
* the full automated test suite passes
* no raw data, processed data, or model binaries are committed
* baseline limitations and results are documented
