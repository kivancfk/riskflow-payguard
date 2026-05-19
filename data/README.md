# Data

Raw Kaggle data is **not** committed to this repository — IEEE-CIS Fraud Detection is large and governed by Kaggle's competition rules.

## Getting the data

1. Accept the competition rules on Kaggle:
   <https://www.kaggle.com/competitions/ieee-fraud-detection>

2. Install the Kaggle CLI and configure your API token (`~/.kaggle/kaggle.json`):

   ```bash
   pip install kaggle
   ```

3. Download into `data/raw/`:

   ```bash
   kaggle competitions download -c ieee-fraud-detection -p data/raw/
   unzip data/raw/ieee-fraud-detection.zip -d data/raw/
   ```

## Layout

```text
data/
├── raw/              # Original Kaggle files (gitignored)
├── processed/        # Cleaned, feature-engineered parquet files (gitignored)
└── sample_payloads/  # Tiny JSON payloads for hitting the API in tests/demos
```

`raw/` and `processed/` are gitignored to keep the repo small and to respect Kaggle's terms. Only `sample_payloads/` is checked in.
