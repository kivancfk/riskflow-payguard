# Phase 3 Calibration and Policy Results

## Status

Phase 3 is complete.

Calibration selection, drift analysis, and policy threshold optimization
used only the chronological validation development partitions. After all
choices were frozen, the resulting artifact was evaluated once against
the untouched chronological test dataset. No component was revised from
the test result.

- Policy version: `calibrated-policy-v1`
- Embedded baseline version: `baseline-v1`
- Local artifact: `models/payguard_calibrated_policy.joblib`
- Test evaluation: complete — final one-time chronological test evaluation
- Artifact creation time: `2026-08-02T20:08:59.200301+00:00`
- Artifact size: `4,306,453` bytes
- Artifact SHA-256:
  `5d53f23719ae891ecc24585393585765aa7fc0900ab38f95e37f59c18fe6c90f`

The artifact is generated locally and remains excluded from Git.

## Validation development partition

The original validation dataset contains 88,581 chronologically ordered
transactions. It was divided by row position into two isolated
development segments.

| Segment | Rows | Transaction IDs | Fraud count | Fraud rate |
|---|---:|---|---:|---:|
| Calibration fit | 44,290 | 3,400,378–3,444,667 | 1,593 | 3.5967% |
| Policy selection | 44,291 | 3,444,668–3,488,958 | 1,449 | 3.2715% |

The selected calibrator was fitted only on `calibration_fit`. Candidate
calibrators and policy thresholds were evaluated only on
`policy_selection`.

## Calibration selection

Candidate selection used the documented lexicographic objective:

1. log loss
2. Brier score
3. expected calibration error
4. calibrator complexity

Metrics were rounded to 12 decimal places for deterministic comparison.

| Method | Log loss | Brier score | ECE | PR-AUC | ROC-AUC | Result |
|---|---:|---:|---:|---:|---:|---|
| Identity | 0.122303 | 0.031577 | 0.037182 | 0.475807 | 0.887798 | Eligible |
| Sigmoid | **0.096791** | 0.022633 | 0.008894 | 0.475807 | 0.887798 | **Selected** |
| Isotonic | 0.098165 | **0.022112** | **0.006544** | 0.462706 | 0.886976 | Eligible |

Sigmoid was selected because it produced the lowest policy-selection
log loss. Isotonic produced slightly lower Brier score and ECE, but the
selection objective prioritizes log loss.

The identical PR-AUC and ROC-AUC values for identity and sigmoid are
expected because sigmoid calibration is a monotonic transformation of
the frozen baseline score. Isotonic calibration introduced tied
probabilities and therefore slightly changed ranking metrics.

## Selected sigmoid calibration metrics

### Calibration-fit segment

| Metric | Value |
|---|---:|
| Transactions | 44,290 |
| Fraud rate | 3.5967% |
| Mean calibrated probability | 3.5965% |
| Log loss | 0.079568 |
| Brier score | 0.018703 |
| ECE | 0.004727 |
| Maximum calibration error | 0.136998 |
| Calibration intercept | 0.000830 |
| Calibration slope | 1.000320 |
| PR-AUC | 0.668864 |
| ROC-AUC | 0.935455 |

The near-zero intercept and near-one slope are expected because this is
the segment on which sigmoid calibration was fitted.

### Policy-selection segment

| Metric | Value |
|---|---:|
| Transactions | 44,291 |
| Fraud rate | 3.2715% |
| Mean calibrated probability | 3.2441% |
| Log loss | 0.096791 |
| Brier score | 0.022633 |
| ECE | 0.008894 |
| Maximum calibration error | 0.262571 |
| Calibration intercept | -0.366700 |
| Calibration slope | 0.844955 |
| PR-AUC | 0.475807 |
| ROC-AUC | 0.887798 |

The later policy-selection segment shows moderate calibration
degradation relative to calibration fit, but the mean calibrated
probability remains close to the observed fraud rate.

## Drift diagnostics

### Score distribution drift

Population Stability Index was calculated from calibration-fit to
policy-selection using reference-quantile bins.

| Score | PSI |
|---|---:|
| Raw baseline score | 0.000635 |
| Calibrated probability | 0.000635 |

Both PSI values are very small, indicating negligible aggregate score
distribution drift between the two adjacent validation periods.

The values are identical because sigmoid calibration is monotonic and
the PSI bins are based on reference quantiles, preserving the segment
counts in corresponding bins.

### Categorical diagnostics

The frozen baseline contains 29 categorical features.

- 25 features had a non-zero missing-rate change.
- 3 features had an increased unknown-category rate.
- Unknown-category changes were very small.
- The largest missing-rate increases were in `M1`, `M2`, and `M3`.

Largest missing-rate changes:

| Feature | Calibration-fit | Policy-selection | Change |
|---|---:|---:|---:|
| M1 | 25.9314% | 27.6332% | +1.7018 pp |
| M2 | 25.9314% | 27.6332% | +1.7018 pp |
| M3 | 25.9314% | 27.6332% | +1.7018 pp |
| M4 | 45.5137% | 46.6438% | +1.1301 pp |
| addr1 | 10.5374% | 9.5572% | -0.9801 pp |
| addr2 | 10.5374% | 9.5572% | -0.9801 pp |

Positive unknown-rate changes:

| Feature | Calibration-fit | Policy-selection | Change |
|---|---:|---:|---:|
| card5 | 0.0045% | 0.0158% | +0.0113 pp |
| addr1 | 0.0135% | 0.0248% | +0.0113 pp |
| card3 | 0.0023% | 0.0068% | +0.0045 pp |

No categorical vocabulary was expanded during these diagnostics.

## Policy assumptions

The policy cost model used the following explicit development
assumptions:

| Assumption | Value |
|---|---:|
| Review cost per transaction | 2.00 |
| Legitimate block cost | 5.00 |
| Fraud loss multiplier | 1.00 |
| Review fraud capture rate | 50% |

These are modeling assumptions for policy development, not measured
production costs.

## Operational constraints

The threshold search required:

| Constraint | Maximum |
|---|---:|
| Review rate | 5.00% |
| Block rate | 1.00% |
| Total intervention rate | 6.00% |

No minimum precision, recall, or fraud-amount-capture constraint was
applied in this development run.

The search evaluated 1,431 threshold pairs. Seven pairs satisfied all
active constraints.

## Selected policy

| Threshold | Value |
|---|---:|
| Review threshold | 0.162551 |
| Block threshold | 0.850922 |

Decision semantics:

- probability below `0.162551`: `ALLOW`
- probability from `0.162551` to below `0.850922`: `REVIEW`
- probability at or above `0.850922`: `BLOCK`

### Policy-selection operating results

| Metric | Value |
|---|---:|
| Transactions | 44,291 |
| Allow count | 42,518 |
| Allow rate | 95.997% |
| Review count | 1,772 |
| Review rate | 4.001% |
| Block count | 1 |
| Block rate | 0.0023% |
| Intervention count | 1,773 |
| Intervention rate | 4.003% |
| Review precision | 42.607% |
| Block precision | 100.000% |
| Fraud intervention recall | 52.174% |
| Expected fraud capture rate | 26.121% |
| Fraud amount capture rate | 22.019% |
| Fraud amount captured | 48,544.9465 |
| Modeled total cost | 175,470.6365 |
| Average modeled cost per transaction | 3.9618 |

The block precision is based on one blocked transaction and should not
be interpreted as a stable production estimate.

## Comparison with all-allow

The all-allow benchmark incurs the full modeled fraud loss.

| Metric | All-allow | Selected policy |
|---|---:|---:|
| Modeled total cost | 220,471.5830 | 175,470.6365 |
| Average cost per transaction | 4.9778 | 3.9618 |
| Prevented fraud loss | 0.0000 | 48,544.9465 |
| Intervention rate | 0.000% | 4.003% |

Selected-policy modeled cost savings:

- Absolute savings: `45,000.9465`
- Cost reduction: `20.4112%`

## Reference threshold comparison

The reference policy used:

- review threshold: `0.30`
- block threshold: `0.70`

Its modeled total cost was `154,029.0715`, which is lower than the
selected feasible policy. However, it blocked 1.4450% of transactions
and therefore violated the maximum 1.00% block-rate constraint.

The optimizer correctly rejected this lower-cost but operationally
infeasible policy.

## Interpretation

The development results support the following conclusions:

1. The weighted baseline probabilities required calibration.
2. Sigmoid calibration materially improved log loss, Brier score, and
   ECE on the later policy-selection period.
3. Aggregate score drift between the two validation periods was
   negligible.
4. Categorical unknown-rate drift was minimal, while several missing
   rates changed modestly.
5. The selected policy meets all configured operational capacity
   constraints.
6. The selected policy improves modeled cost relative to all-allow, but
   its economic results depend directly on the stated cost assumptions.
7. The selected thresholds and calibrator are frozen before final test
   evaluation.

## Final one-time chronological test evaluation

The frozen `calibrated-policy-v1` artifact was evaluated once after all
calibration and threshold decisions had been recorded.

The SHA-256 hash was identical before and after evaluation:

`5d53f23719ae891ecc24585393585765aa7fc0900ab38f95e37f59c18fe6c90f`

No calibrator, threshold, cost assumption, constraint, feature contract,
or model parameter was revised from the test result.

### Test split

| Field | Value |
|---|---:|
| Transactions | 88,581 |
| Fraud count | 3,083 |
| Fraud rate | 3.4804% |
| First TransactionID | 3,488,959 |
| Last TransactionID | 3,577,539 |

### Raw and calibrated probability performance

| Metric | Raw baseline | Sigmoid calibrated |
|---|---:|---:|
| PR-AUC | 0.494612 | 0.494612 |
| ROC-AUC | 0.880848 | 0.880848 |
| Log loss | 0.130046 | **0.101439** |
| Brier score | 0.033580 | **0.023792** |

Sigmoid calibration improved probability accuracy while preserving
ranking performance.

The mean calibrated probability was `3.4883%`, close to the observed
fraud rate of `3.4804%`.

### Calibration diagnostics

| Metric | Value |
|---|---:|
| Expected calibration error | 0.010240 |
| Maximum calibration error | 0.210560 |
| Calibration intercept | -0.395917 |
| Calibration slope | 0.832946 |
| Minimum calibrated probability | 0.006789 |
| Maximum calibrated probability | 0.850920 |

The intercept and slope indicate some temporal calibration degradation,
but overall probability levels remained close to observed prevalence.

### Review-capacity performance

| Review capacity | Fraud recall | Review precision | Fraud amount capture |
|---|---:|---:|---:|
| 0.5% | 12.812% | 89.165% | 6.438% |
| 1.0% | 24.100% | 83.860% | 14.158% |
| 2.0% | 39.572% | 68.849% | 26.955% |
| 5.0% | 56.017% | 38.984% | 44.174% |

### Frozen policy performance

The frozen thresholds were:

- review threshold: `0.16255069862369795`
- block threshold: `0.8509223095305902`

| Metric | Test result |
|---|---:|
| Allow count | 84,646 |
| Allow rate | 95.558% |
| Review count | 3,935 |
| Review rate | 4.442% |
| Block count | 0 |
| Block rate | 0.000% |
| Total intervention rate | 4.442% |
| Review fraud count | 1,672 |
| Review precision | 42.490% |
| Fraud intervention recall | 54.233% |
| Expected fraud capture rate | 27.116% |
| Fraud amount capture rate | 21.127% |
| Fraud amount captured | 99,216.3350 |
| Modeled total cost | 378,262.1860 |
| Average modeled cost per transaction | 4.2702 |

The policy satisfied all frozen operational constraints, with no
constraint violations.

### Block-threshold observation

The maximum calibrated test probability was
`0.8509198705494854`.

This was slightly below the frozen block threshold of
`0.8509223095305902`.

Therefore, no test transactions received a `BLOCK` decision. During the
test period, the policy operated as an `ALLOW` or `REVIEW` policy.

The block threshold must not be lowered based on this test result. Any
future threshold revision requires a new policy version and new
validation evidence.

### Comparison with all-allow

| Metric | All-allow | Frozen policy |
|---|---:|---:|
| Modeled total cost | 469,608.5210 | 378,262.1860 |
| Average cost per transaction | 5.3015 | 4.2702 |
| Prevented fraud loss | 0.0000 | 99,216.3350 |
| Intervention rate | 0.000% | 4.442% |

Under the frozen development assumptions:

- modeled cost savings: `91,346.3350`
- modeled cost reduction: `19.4516%`

These are scenario results based on assumed costs and a 50% review fraud
capture rate. They are not measured production savings.

### Development-to-test stability

| Metric | Policy selection | Final test |
|---|---:|---:|
| Review rate | 4.001% | 4.442% |
| Block rate | 0.002% | 0.000% |
| Review precision | 42.607% | 42.490% |
| Fraud intervention recall | 52.174% | 54.233% |
| Fraud amount capture | 22.019% | 21.127% |
| Modeled cost reduction | 20.411% | 19.452% |

The frozen policy generalized consistently from policy selection to the
chronological test period.

## Phase 3 conclusion

Phase 3 delivered:

- chronological calibration and policy partitions
- identity, sigmoid, and isotonic calibration candidates
- deterministic sigmoid selection
- probability and categorical drift diagnostics
- explicit policy costs and operational constraints
- deterministic `ALLOW`, `REVIEW`, and `BLOCK` threshold optimization
- a versioned and atomically persisted policy bundle
- raw-feature calibrated-policy inference
- a final one-time chronological test evaluation

The resulting policy is a technical and portfolio benchmark, not a
production payment-authorization system.
