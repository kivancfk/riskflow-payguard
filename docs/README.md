# Documentation

Long-form project documentation, kept out of the root README so the front page
stays scannable.

## Implemented Phase Documentation

- [`phase1_data_eda.md`](phase1_data_eda.md) — data setup, EDA, feature engineering, and chronological datasets
- [`phase2_baseline_model.md`](phase2_baseline_model.md) — reproducible LightGBM baseline implementation
- [`phase2_baseline_results.md`](phase2_baseline_results.md) — frozen baseline evaluation results
- [`phase3_calibration_thresholds.md`](phase3_calibration_thresholds.md) — calibration and deterministic policy-selection design
- [`phase3_results.md`](phase3_results.md) — frozen calibrated-policy results
- [`phase4_explanations_reason_codes.md`](phase4_explanations_reason_codes.md) — TreeSHAP explanations and analyst reason codes
- [`phase4_results.md`](phase4_results.md) — Phase 4 verification and explanation results
- [`phase5_api_integration.md`](phase5_api_integration.md) — strict FastAPI inference contract and parity verification
- [`phase6_monitoring_dashboard.md`](phase6_monitoring_dashboard.md) — prediction persistence, outcome labels, monitoring, and read-only threshold simulation
- [`phase7_deployment.md`](phase7_deployment.md) — hardened container image, PostgreSQL Compose deployment, persistence portability, and smoke validation
- [`phase8_portfolio_demo.md`](phase8_portfolio_demo.md) — completed API-backed Product Demo, captured portfolio screenshots, optional video plan, and product boundaries

## Current Portfolio Surface

The root [`README.md`](../README.md) is the primary project landing page.

Phase 8 adds a guided Streamlit Product Demo over the existing FastAPI
inference path. The demonstration is designed to show:

~~~text
transaction
    |
    v
FastAPI scoring
    |
    v
frozen decision + explanation
    |
    v
prediction persistence
    |
    v
Monitoring
    |
    v
read-only Threshold Simulator
~~~

The detailed presentation workflow, captured screenshot set, and optional
portfolio-video plan are documented in
[`phase8_portfolio_demo.md`](phase8_portfolio_demo.md).

## Future Documentation

Later work should document capabilities only when they are actually
implemented, including:

- public-cloud release;
- monitoring alerts;
- authentication and authorization;
- rate limiting and production traffic controls;
- automated ground-truth ingestion;
- database migrations for schema evolution;
- backup and restore procedures;
- model-risk and policy-approval governance;
- separately versioned future model or policy releases.

The current documentation should not imply that these future capabilities
already exist.
