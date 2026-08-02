# Phase 1 — Data Setup and Fraud-Risk EDA

## Objective

Phase 1 establishes a reproducible local data workflow and examines the IEEE-CIS Fraud Detection dataset from the perspective of a fraud-risk product.

The purpose is not to produce every possible chart. The EDA should answer the questions needed to design:

* a reliable fraud-scoring dataset
* a practical baseline model
* an API-compatible feature contract
* future fraud-monitoring dashboards
* safe chronological model evaluation

## Source data

The project uses the IEEE-CIS Fraud Detection dataset.

### Main tables

| Table                   | Purpose                                 | Expected grain                      |
| ----------------------- | --------------------------------------- | ----------------------------------- |
| `train_transaction.csv` | Main labeled transaction dataset        | One row per `TransactionID`         |
| `train_identity.csv`    | Optional identity and device attributes | Zero or one row per `TransactionID` |
| `test_transaction.csv`  | Unlabeled Kaggle test transactions      | One row per `TransactionID`         |
| `test_identity.csv`     | Optional test identity attributes       | Zero or one row per `TransactionID` |
| `sample_submission.csv` | Kaggle submission structure             | One row per test `TransactionID`    |

### Join contract

* Primary key: `TransactionID`
* Primary table: transaction
* Join type: left join
* Relationship: one transaction to zero or one identity row
* The transaction row count must not change after joining identity data.

Only a subset of transactions has an identity record. Missing identity data must therefore not automatically be treated as invalid data.

### Schema normalization

The train identity table uses names such as:

```text
id_01
id_02
```

The test identity table uses names such as:

```text
id-01
id-02
```

The data loader normalizes test identity names to the underscore convention.

## Key fields

### Target

```text
isFraud
```

Interpretation:

* `0`: legitimate transaction
* `1`: fraudulent transaction

The target exists only in the training transaction table.

### Time field

```text
TransactionDT
```

`TransactionDT` is an elapsed-time value rather than a business timestamp.

It should be used for:

* chronological ordering
* relative day and hour features
* time-based fraud-rate analysis
* chronological train, validation, and test splits

It should not be presented as an exact calendar timestamp unless a justified reference date is introduced.

### Transaction amount

```text
TransactionAmt
```

This is the primary monetary feature.

EDA should examine:

* distribution and skew
* percentiles
* extreme values
* fraud rate by amount band
* fraud amount share
* differences by product and payment attributes

## Feature groups

The EDA should organize features into meaningful risk domains rather than inspecting hundreds of columns individually.

### Transaction context

Examples:

* `TransactionAmt`
* `ProductCD`
* `TransactionDT`

Questions:

* Which product groups have the highest fraud rate?
* Are fraudulent transactions concentrated in particular amount bands?
* Does fraud behavior change over time?

### Card and payment attributes

Examples:

* `card1`–`card6`

Questions:

* Which card-related fields behave like identifiers?
* Which fields are categorical despite numeric storage?
* Are fraud rates concentrated in specific card categories?
* Which fields have very high cardinality?

### Address and distance signals

Examples:

* `addr1`
* `addr2`
* `dist1`
* `dist2`

Questions:

* How much data is missing?
* Does the presence or absence of distance information relate to fraud?
* Are extreme distance values associated with higher fraud rates?

### Email-domain signals

Examples:

* `P_emaildomain`
* `R_emaildomain`

Questions:

* How common is each domain?
* Which domains have enough transactions for reliable fraud-rate estimates?
* Does payer and recipient domain mismatch provide a useful signal?
* Should rare domains be grouped before modeling?

### Count and frequency signals

Examples:

* `C1`–`C14`

These fields may represent transaction-count or behavioral-frequency information.

Questions:

* Which count variables are highly skewed?
* Which have useful separation between fraud and non-fraud?
* Are transformations such as `log1p` appropriate?

### Time-delta signals

Examples:

* `D1`–`D15`

Questions:

* Which fields contain substantial missingness?
* Does missingness itself carry fraud signal?
* Which values behave like elapsed-time or account-age indicators?
* Which fields may need clipping or normalization?

### Match indicators

Examples:

* `M1`–`M9`

Questions:

* What categories exist?
* How common is missingness?
* Which match states show different fraud rates?

### Anonymous engineered variables

Examples:

* `V1`–`V339`

These variables should initially be treated as anonymous numerical risk signals.

Phase 1 should focus on:

* missingness
* constant or near-constant columns
* numerical ranges
* correlations
* leakage risk
* memory requirements

The first baseline should not attempt extensive manual interpretation of every `V` variable.

### Identity and device attributes

Examples:

* `DeviceType`
* `DeviceInfo`
* `id_01`–`id_38`

Questions:

* What percentage of transactions have identity data?
* Does identity-data availability correlate with the target?
* Which device and browser fields have manageable cardinality?
* Which identity fields are too sparse for the first baseline?

## Required EDA outputs

### 1. Dataset dimensions

Record:

* transaction row count
* identity row count
* column count
* joined row count
* percentage of transactions with identity records
* uniqueness of `TransactionID`

### 2. Target imbalance

Report:

* legitimate transaction count
* fraudulent transaction count
* overall fraud rate
* legitimate-to-fraud ratio

This determines:

* suitable evaluation metrics
* class-weight strategy
* threshold design
* whether resampling is needed

Accuracy should not be used as the primary metric.

### 3. Missingness

For every feature, calculate:

* missing count
* missing percentage
* data type
* number of distinct values

Investigate:

* columns with more than 90% missingness
* columns with 50%–90% missingness
* missingness differences between fraud and legitimate transactions
* identity-data availability as a possible feature

Missing values should not automatically be filled during EDA.

### 4. Cardinality

For categorical or identifier-like fields, calculate:

* unique values
* unique-value ratio
* most frequent values
* rare-category frequency
* fraud rate for sufficiently large categories

Classify fields as:

* low cardinality
* medium cardinality
* high cardinality
* identifier-like

High-cardinality columns should not be one-hot encoded without review.

### 5. Transaction amount behavior

Analyze:

* minimum
* median
* mean
* upper percentiles
* maximum
* skewness
* fraud rate by quantile or business-readable amount band
* transaction-value share associated with fraud

A logarithmic view may be used for visualization, but the raw amount must be preserved.

### 6. Time behavior

Analyze fraud by relative:

* day
* week
* hour-of-day
* chronological period

Check:

* transaction volume over time
* fraud-rate drift
* changes in feature availability
* whether later periods differ materially from earlier periods

The model split must be chronological rather than random.

### 7. Fraud-rate patterns

For selected product-risk dimensions, report:

* transaction count
* fraud count
* fraud rate
* transaction amount
* fraudulent transaction amount

Initial dimensions:

* `ProductCD`
* `card4`
* `card6`
* `P_emaildomain`
* `R_emaildomain`
* `DeviceType`
* identity-data availability
* transaction-amount band
* relative time period

Fraud rates for very small groups should not be interpreted without minimum-support thresholds.

### 8. Data-quality checks

Check for:

* duplicate `TransactionID` values
* changed row count after joining
* missing target values
* invalid target values
* negative transaction amounts
* infinite numerical values
* constant columns
* train/test schema differences
* suspicious target leakage

## Baseline model decision

The first baseline should be a compact LightGBM classifier.

The baseline should prioritize:

* fast iteration
* mixed numerical and categorical inputs
* missing-value tolerance
* nonlinear fraud patterns
* class imbalance handling
* probability output for risk thresholds

### Initial baseline feature scope

The first baseline should use a controlled feature set rather than all available columns.

Include:

* `TransactionAmt`
* relative day
* relative hour
* `ProductCD`
* `card1`–`card6`
* `addr1`
* `addr2`
* `dist1`
* `dist2`
* `P_emaildomain`
* `R_emaildomain`
* selected `C` variables
* selected `D` variables
* selected `M` variables
* identity-data availability
* `DeviceType`
* selected low-missingness identity fields

Defer initially:

* all `V1`–`V339` variables
* raw `DeviceInfo`
* uncontrolled high-cardinality text
* target encoding
* behavioral aggregations requiring historical windows
* SMOTE
* probability calibration
* SHAP-based explanations

These may be introduced after the compact baseline is validated.

## Validation strategy

Use a chronological split based on `TransactionDT`.

Initial proportions:

* training: earliest 70%
* validation: next 15%
* test: latest 15%

The Kaggle test dataset should not be used as the internal model test set because it does not contain labels.

### Primary evaluation metrics

Use:

* ROC AUC
* PR AUC
* log loss
* Brier score
* recall at selected precision
* precision and recall at decision thresholds

Later product evaluation should also include:

* approval rate
* manual-review rate
* block rate
* fraud capture rate
* false-positive rate
* transaction-value exposure

## Phase 1 processed outputs

Phase 1 should eventually produce local, gitignored artifacts such as:

```text
data/processed/
├── baseline_dataset.parquet
├── train.parquet
├── validation.parquet
├── test.parquet
├── feature_metadata.csv
└── eda_summary.json
```

These names may be adjusted when the notebook and feature contract are finalized.

## Phase 1 completion criteria

Phase 1 is complete when:

* local data download is reproducible
* raw data remains excluded from Git
* train and test schemas are normalized
* transaction and identity joins are validated
* target imbalance is documented
* missingness and cardinality are summarized
* time and amount behavior are analyzed
* fraud-rate patterns are documented
* a compact baseline feature set is selected
* chronological split logic is defined
* processed baseline data can be reproduced
