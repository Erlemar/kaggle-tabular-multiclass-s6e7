# EDA Report

This document outlines the findings of our initial exploratory data analysis on the playground-series-s6e7 dataset.

## General Overview

- **Training shape:** (690088, 15)
- **Test shape:** (295753, 14)
- **Metric:** accuracy
- **Target column:** health_condition
- **Classes (3):** at-risk, fit, unhealthy

## Target Distribution

There is a significant class imbalance (Imbalance ratio ~ 14.9x).

- **at-risk:** 592,561 (Majority class)
- **unhealthy:** 57,724
- **fit:** 39,803

The fold scheme to use must be **stratified** to preserve this class balance per fold.

## Missing Values

Several columns contain missing values, though the rates are relatively low (max 12%). We should consider adding missing indicator flags for informative features.

- `stress_level`: 12.00%
- `sleep_duration`: 11.01%
- `sleep_quality`: 8.45%
- `calorie_expenditure`: 7.66%
- `water_intake`: 6.30%
- `physical_activity_level`: 5.31%
- `smoking_alcohol`: 4.14%
- `gender`: 3.10%

## Features

- **Numeric columns (7):** `sleep_duration`, `heart_rate`, `bmi`, `calorie_expenditure`, `step_count`, `exercise_duration`, `water_intake`.
- **Categorical columns (6):** `diet_type`, `stress_level`, `sleep_quality`, `physical_activity_level`, `smoking_alcohol`, `gender`. All are low cardinality (3 unique values each). One-hot encoding or native handling by CatBoost is recommended.

## Potential Issues

### 1. Label Noise / Duplicates
- **Duplicate rows:** 0
- **Duplicate conflict rate:** 0.0%
No concerns here.

### 2. Adversarial Validation (Train/Test Drift)
- **AUC:** ~0.654
This indicates a **moderate train/test shift** (ideally it should be ~0.5).
**Top drift features:** `water_intake`, `exercise_duration`, `step_count`, `heart_rate`, `calorie_expenditure`, `sleep_duration`, `bmi`, `diet_type`. 
*Note:* We must be cautious with the shifting features as models over-relying on them could face degraded LB performance despite solid CV scores.

### 3. Leakage Suspects
Features with the highest mutual information to the target:
- `sleep_duration`: 0.1551
- `bmi`: 0.0323
- `exercise_duration`: 0.0228
None of these MI values are extremely high (>0.8) to suggest guaranteed leakage, but they are clearly the strongest predictors.

## Recommendations for Feature Engineering

1. Low cardinality categoricals can be One-Hot Encoded or fed raw to CatBoost/LightGBM.
2. Consider missing-indicator columns since missingness might be correlated with target class.
3. Group aggregations (mean/std/min/max/count) of numerics grouped by categorical columns (`gender`, `stress_level`, `diet_type`).
4. Consider domain ratios or differences, e.g., `step_count / calorie_expenditure`.
