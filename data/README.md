# Data

RiskFlow PayGuard uses the IEEE-CIS Fraud Detection dataset from Kaggle.

The raw competition files are downloaded locally and must not be committed to this repository.

## Directory contract

```text
data/
├── raw/
│   ├── train_transaction.csv
│   ├── train_identity.csv
│   ├── test_transaction.csv
│   ├── test_identity.csv
│   └── sample_submission.csv
├── processed/
│   └── Generated analytical and model-ready Parquet files
└── sample_payloads/
    └── Small synthetic JSON payloads for API tests and demonstrations
```

## Raw data

`data/raw/` contains the original Kaggle files.

Rules:

* Treat raw files as immutable inputs.
* Do not manually edit them.
* Do not commit them.
* Recreate them using the download script when needed.

## Processed data

`data/processed/` contains generated outputs such as:

* joined transaction and identity datasets
* reduced EDA datasets
* model-ready feature tables
* train, validation, and test datasets
* feature and schema metadata

Parquet is preferred because it preserves data types and supports more efficient analytical reads than CSV.

Processed files are also excluded from Git.

## Sample payloads

`data/sample_payloads/` contains small synthetic examples for:

* FastAPI request testing
* automated tests
* dashboard demonstrations
* API documentation

Real Kaggle transaction rows must not be copied into committed payload files.

## Kaggle setup

Before downloading the dataset:

1. Join the IEEE-CIS Fraud Detection competition on Kaggle.
2. Accept the competition rules.
3. Download your Kaggle API token.
4. Store the token at:

```text
~/.kaggle/kaggle.json
```

On macOS or Linux, restrict the token permissions:

```bash
chmod 600 ~/.kaggle/kaggle.json
```

The Kaggle token is a local credential and must never be committed to Git.

## Install development dependencies

Activate the project virtual environment:

```bash
source .venv/bin/activate
```

Install the local development dependencies:

```bash
python -m pip install -r requirements-dev.txt
```

These dependencies include:

* Kaggle CLI for dataset access
* JupyterLab and IPython kernel for EDA
* PyArrow for Parquet output

## Download the dataset

From the repository root, run:

```bash
python scripts/download_ieee_cis.py
```

The script:

1. Downloads the IEEE-CIS Fraud Detection competition archive.
2. Extracts the five expected CSV files into `data/raw/`.
3. Verifies that all expected files exist.
4. Rejects missing or empty files.
5. Removes the downloaded ZIP archive after successful extraction.

The expected raw files are:

```text
data/raw/train_transaction.csv
data/raw/train_identity.csv
data/raw/test_transaction.csv
data/raw/test_identity.csv
data/raw/sample_submission.csv
```

To keep the downloaded ZIP archive:

```bash
python scripts/download_ieee_cis.py --keep-archive
```

To use a different raw-data directory:

```bash
python scripts/download_ieee_cis.py --raw-dir /path/to/raw-data
```

## Verify the local dataset

After downloading, inspect the files:

```bash
ls -lh data/raw/
```

The downloader also reports the size of every validated CSV file.

## Verify Git protection

Check that a raw file is ignored:

```bash
git check-ignore data/raw/train_transaction.csv
```

Expected output:

```text
data/raw/train_transaction.csv
```

Check that processed outputs are also ignored:

```bash
git check-ignore data/processed/example.parquet
```

Expected output:

```text
data/processed/example.parquet
```

Finally, inspect the working tree:

```bash
git status --short
```

Kaggle CSV, ZIP, Parquet, and credential files must not appear in Git status.
## Phase 1 processed datasets

After downloading the IEEE-CIS source files, generate the compact baseline
datasets from the repository root:

```bash
python scripts/materialize_phase1.py
```

The workflow:

1. validates and loads the raw transaction and identity tables
2. builds the deterministic 63-feature baseline contract
3. uses the minimum training `TransactionDT` as the shared time origin
4. creates stable chronological 70%/15%/15% labeled splits
5. processes the unlabeled Kaggle test set using the same feature contract
6. validates each artifact before moving it into its final location

Generated files:

```text
data/processed/
├── train.parquet
├── validation.parquet
├── test.parquet
├── kaggle_test.parquet
├── feature_metadata.csv
└── dataset_manifest.json
```

Internal labeled Parquet files contain:

- `TransactionID`
- `isFraud`
- 63 baseline model features

`kaggle_test.parquet` contains:

- `TransactionID`
- 63 baseline model features
- no target column

The manifest records:

- chronological split boundaries and fraud rates
- identity coverage
- transaction-value summaries
- categorical and numerical feature lists
- output schemas and artifact sizes
- the shared relative-time origin

Processed artifacts are local generated data and are excluded from Git.

### Rebuilding artifacts

Existing outputs are protected from accidental replacement. To rebuild them
intentionally:

```bash
python scripts/materialize_phase1.py --overwrite
```

Alternative locations can be supplied when needed:

```bash
python scripts/materialize_phase1.py \
  --raw-dir /path/to/raw \
  --processed-dir /path/to/processed
```

The materializer can also be invoked as a module:

```bash
python -m scripts.materialize_phase1
```
