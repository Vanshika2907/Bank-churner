# Data Quality Validation Report

## Dataset overview
- Source file: `ml_ready_dataset.csv`
- Dataset shape: 10000 rows × 32 columns
- Duplicate customers: 0

## Target distribution
- Class counts: {0: 4488, 1: 5512}
- Target prevalence: 0.5512

## Missing values
- Missing entries total: 0
- Fields with missing values: None

## Major correlations (by absolute Pearson coefficient)
- transaction_count vs number_of_active_products: 0.988
- number_of_active_products vs transaction_count: 0.988
- digital_banking_activity_level vs number_of_active_products: 0.9584
- number_of_active_products vs digital_banking_activity_level: 0.9584
- transaction_count vs digital_banking_activity_level: 0.9496
- digital_banking_activity_level vs transaction_count: 0.9496
- average_transaction_value vs number_of_active_products: 0.8905
- number_of_active_products vs average_transaction_value: 0.8905
- complaint_occurrence vs complaint_resolution_time: 0.8857
- complaint_resolution_time vs complaint_occurrence: 0.8857

## Leakage risk assessment
- The data uses only historical observation-window behavior to define the features.
- The target is derived from a latent risk process using the same observation period and a future outcome window, which is valid for the design.
- The pipeline should still be checked before modeling to confirm no future-period variables are accidentally merged into features.

## Synthetic realism assessment
- The synthetic data is intentionally correlated across financial, transactional, engagement, and service-friction dimensions.
- It includes marginal heterogeneity, moderate noise, and realistic class overlap to avoid an unrealistically easy target.
- The generated data is suitable for method validation, reproducible experimentation, and initial model prototyping.

## Recommended next step
- Run the ETL validation and feature engineering notebook pipeline.
- Build and evaluate the baseline classifier only after the dataset is confirmed to be leakage-safe and stable.