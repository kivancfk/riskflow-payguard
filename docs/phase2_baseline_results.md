# Phase 2 — Baseline Model Results

## Summary

RiskFlow PayGuard Phase 2 produced the first reproducible LightGBM fraud-risk baseline using the chronological IEEE-CIS Fraud Detection datasets materialized in Phase 1.

The frozen model was trained on the training split, used the validation split for early stopping, and was evaluated once against the untouched chronological test split.

The baseline demonstrates meaningful fraud-ranking capability:

* validation PR-AUC: `0.5772`
* test PR-AUC: `0.4946`
* test ROC-AUC: `0.8808`
* top 1% test review queue:

  * fraud recall: `24.10%`
  * review precision: `83.86%`
  * fraudulent amount captured: `14.16%`

The test PR-AUC is approximately 14.2 times the test fraud prevalence of 3.48%. The model is therefore a useful technical baseline, but it is not yet a calibrated or production-ready decision system.

## Run metadata

| Field                 |                                      Value |
| --------------------- | -----------------------------------------: |
| Run date              |                                 2026-08-02 |
| Source commit         | `056eabc920aa9d7a6efd1951b351380b351c4c8e` |
| Model version         |                              `baseline-v1` |
| Bundle schema version |                                        `2` |
| Training duration     |                                 91 seconds |
| Artifact path         |          `models/payguard_baseline.joblib` |
| Artifact size         |    4,298,068 bytes, approximately 4.10 MiB |
| Best iteration        |                                      1,454 |
| Maximum estimators    |                                      2,000 |
| Positive-class weight |                                    27.4343 |
| Feature count         |                                         63 |
| Categorical features  |                                         29 |
| Numerical features    |                                         34 |
| Random seed           |                                         42 |
| Training threads      |                                          4 |

The model artifact, processed datasets, run log, and machine-generated JSON report remain local and are excluded from Git.

## Dataset policy

The Phase 1 chronological split was preserved:

| Split      | Transactions | Fraud transactions |          Fraud rate |
| ---------- | -----------: | -----------------: | ------------------: |
| Training   |      413,378 |                  — | approximately 3.52% |
| Validation |       88,581 |              3,042 |             3.4341% |
| Test       |       88,581 |              3,083 |             3.4804% |

The workflow followed these rules:

1. Categorical vocabularies were learned from training data only.
2. Model parameters were fitted on the training split.
3. The validation split was used for early stopping.
4. Model code and parameters were frozen before test evaluation.
5. The test split was evaluated once for the final Phase 2 baseline.
6. The Kaggle competition test data was not used for internal evaluation.

No rows were randomly redistributed between the chronological splits.

## Model configuration

The baseline used `LGBMClassifier` with a binary objective and deterministic CPU configuration.

| Parameter               |   Value |
| ----------------------- | ------: |
| `learning_rate`         |    0.03 |
| `n_estimators`          |   2,000 |
| `num_leaves`            |      31 |
| `max_depth`             |      -1 |
| `min_child_samples`     |     100 |
| `subsample`             |    0.90 |
| `subsample_freq`        |       1 |
| `colsample_bytree`      |    0.90 |
| `reg_alpha`             |    0.10 |
| `reg_lambda`            |    1.00 |
| `early_stopping_rounds` |     100 |
| `seed`                  |      42 |
| `scale_pos_weight`      | 27.4343 |
| `deterministic`         |    true |
| `force_col_wise`        |    true |

The positive-class weight was calculated from the training split as:

```text
negative training transactions / fraudulent training transactions
```

No oversampling, SMOTE, numerical scaling, or numerical-value imputation was applied. LightGBM handled numerical missing values natively.

## Probability and ranking metrics

| Metric           | Validation |   Test | Test change |
| ---------------- | ---------: | -----: | ----------: |
| Fraud prevalence |     0.0343 | 0.0348 |     +0.0005 |
| PR-AUC           |     0.5772 | 0.4946 |     -0.0826 |
| ROC-AUC          |     0.9128 | 0.8808 |     -0.0319 |
| Log loss         |     0.1116 | 0.1300 |     +0.0185 |
| Brier score      |     0.0290 | 0.0336 |     +0.0046 |

Test PR-AUC declined by approximately 14.3% relative to validation. ROC-AUC declined by approximately 3.5%, while log loss and Brier score increased by approximately 16.5% and 15.9%, respectively.

This degradation indicates temporal distribution shift between validation and test periods. The test result remains substantially stronger than random or prevalence-only ranking, but future phases should investigate drift, probability calibration, and stability across time windows.

## Reference threshold metrics

The `0.50` threshold is included only as a diagnostic reference. It is not the final PayGuard allow, review, or block threshold.

| Metric at 0.50               | Validation |   Test |
| ---------------------------- | ---------: | -----: |
| Precision                    |     49.13% | 42.74% |
| Recall                       |     59.70% | 54.14% |
| F1 score                     |     53.90% | 47.77% |
| False-positive rate          |      2.20% |  2.62% |
| False-negative rate          |     40.30% | 45.86% |
| Predicted fraud transactions |      3,696 |  3,905 |
| True positives               |      1,816 |  1,669 |
| False positives              |      1,880 |  2,236 |
| False negatives              |      1,226 |  1,414 |
| True negatives               |     83,659 | 83,262 |

The class-weighted model probabilities are not calibrated, so a `0.50` score should not be interpreted as a literal 50% fraud probability.

## Review-capacity results

### Validation

| Review capacity | Reviewed | Fraud found | Fraud recall | Review precision | Fraud amount captured |
| --------------- | -------: | ----------: | -----------: | ---------------: | --------------------: |
| Top 0.5%        |      443 |         412 |       13.54% |           93.00% |                 9.53% |
| Top 1%          |      886 |         799 |       26.27% |           90.18% |                20.09% |
| Top 2%          |    1,772 |       1,364 |       44.84% |           76.98% |                38.35% |
| Top 5%          |    4,430 |       1,905 |       62.62% |           43.00% |                55.25% |

### Test

| Review capacity | Reviewed | Fraud found | Fraud recall | Review precision | Fraud amount captured |
| --------------- | -------: | ----------: | -----------: | ---------------: | --------------------: |
| Top 0.5%        |      443 |         395 |       12.81% |           89.16% |                 6.44% |
| Top 1%          |      886 |         743 |       24.10% |           83.86% |                14.16% |
| Top 2%          |    1,772 |       1,220 |       39.57% |           68.85% |                26.96% |
| Top 5%          |    4,430 |       1,727 |       56.02% |           38.98% |                44.17% |

The top 1% test queue contains fraud at approximately 24 times the underlying test prevalence:

```text
83.86% review precision / 3.48% fraud prevalence ≈ 24.1
```

This demonstrates that the model can create a highly concentrated manual-review queue. Fraudulent amount recall is lower than transaction-level fraud recall, suggesting that ranking performance differs for high-value fraudulent transactions and should be investigated separately.

## Feature importance

The ten highest features by LightGBM gain importance were:

| Rank | Feature         |      Gain |
| ---: | --------------- | --------: |
|    1 | `card1`         | 6,517,646 |
|    2 | `card2`         | 1,038,104 |
|    3 | `addr1`         |   854,176 |
|    4 | `R_emaildomain` |   794,314 |
|    5 | `C1`            |   723,087 |
|    6 | `C5`            |   661,191 |
|    7 | `C14`           |   579,522 |
|    8 | `D3`            |   485,388 |
|    9 | `id_31`         |   429,891 |
|   10 | `D2`            |   332,244 |

Gain importance indicates how much features contributed to tree splits. It is not a causal interpretation and does not explain individual predictions. High-cardinality features may also receive disproportionate importance.

SHAP-based local and global explanations are deferred to a later phase.

## Artifact contents

The versioned joblib bundle contains:

* fitted LightGBM classifier
* categorical encoder
* ordered feature contract
* categorical vocabularies
* training configuration
* positive-class weight
* best iteration
* validation metrics
* final test metrics
* model version
* creation timestamp
* source dataset manifest

The bundle can be loaded without access to the training DataFrames and can score compatible raw feature frames using the same categorical mappings used during training.

Atomic persistence, overwrite protection, bundle validation, reloading, and post-load prediction are covered by automated tests.

## Known limitations

1. **Probability calibration**

   Positive-class weighting improves minority-class learning but produces probabilities that should not be treated as calibrated fraud likelihoods. Calibration is required before probability-based business interpretation.

2. **Reference threshold only**

   The `0.50` threshold is diagnostic. Review and block thresholds must be chosen using business costs, operational capacity, false-positive tolerance, and fraud-loss objectives.

3. **Temporal performance decline**

   Test performance is lower than validation performance, especially for PR-AUC and fraudulent amount capture. This indicates temporal drift or changing relationships between features and fraud.

4. **No hyperparameter optimization**

   The parameters were selected as a defensible baseline. No grid search, Bayesian optimization, or repeated test-guided tuning was performed.

5. **No probability or threshold optimization by transaction value**

   The baseline ranks fraud transactions but does not explicitly optimize expected monetary loss or high-value fraud capture.

6. **No individual explanations**

   Gain importance is global and model-level. SHAP explanations and reason codes have not yet been implemented.

7. **No production decision engine integration**

   The model bundle is not yet connected to the FastAPI prediction endpoint or the allow, review, and block decision workflow.

8. **No monitoring framework**

   Feature drift, prediction drift, category novelty, calibration decay, and realized fraud outcomes are not yet monitored.

9. **Single final test observation**

   The chronological test split was evaluated once after freezing the implementation. Further model decisions should not be made by repeatedly inspecting this same test result.

## Baseline conclusion

The Phase 2 LightGBM model is accepted as the first RiskFlow PayGuard technical baseline.

It satisfies the principal Phase 2 objectives:

* reproducible chronological training
* training-only categorical preprocessing
* class-imbalance handling
* validation-based early stopping
* fraud-focused evaluation
* fixed-capacity review metrics
* one-time chronological test evaluation
* versioned model persistence
* artifact reloading and inference
* automated test coverage

The baseline is not production-ready. The most important future improvements are probability calibration, temporal-drift analysis, business threshold optimization, SHAP explanations, and API integration.

No additional tuning should be performed against the current test split. Future model comparisons should preserve the recorded `baseline-v1` results as the benchmark.
