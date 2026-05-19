"""Train a LightGBM fraud classifier and save the artifact.

Usage:
    python -m src.train

TODO:
    - load processed data
    - class imbalance: scale_pos_weight or SMOTE
    - LightGBM with early stopping on validation set
    - probability calibration (see src.calibration)
    - SHAP TreeExplainer fitted once
    - persist {model, explainer, feature_names, model_version, metrics} via joblib
"""
from pathlib import Path

import joblib


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
