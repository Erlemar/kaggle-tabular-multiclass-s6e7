# Engineered Features Report

This document records the features added to `features_train.parquet` and `features_test.parquet` based on the exploratory data analysis findings.

## Methodology
The script (`src/features.py`) applies transformations out-of-fold where necessary to avoid leakage. It preserves id alignment across the dataset.

## Added Features

1. **Missing Indicators:**
   - Appended a `{col}_isnull` indicator column (0/1) for every feature that had missing values in the train/test sets, as missingness might be correlated with the target class.

2. **Categorical Features (Native Categories):**
   - Transformed all 6 low-cardinality categorical variables (`diet_type`, `stress_level`, `sleep_quality`, `physical_activity_level`, `smoking_alcohol`, `gender`) into `category` dtypes for seamless native handling by tree-based models like CatBoost or LightGBM. Null values were mapped to "Missing".

3. **Frequency/Count Encoding:**
   - Added `{col}_count` features for each categorical variable. This maps categories to their relative frequency count (across both train and test data), giving tree models a hint of class popularity without risking leakage.

4. **Numeric Ratios:**
   - Evaluated interacting numeric columns to create informative domain ratios:
     - `ratio_step_calorie`: step_count / calorie_expenditure
     - `ratio_step_bmi`: step_count / bmi
     - `ratio_exercise_calorie`: exercise_duration / calorie_expenditure
     - `ratio_sleep_exercise`: sleep_duration / exercise_duration
     - `ratio_water_exercise`: water_intake / exercise_duration
     - `ratio_water_bmi`: water_intake / bmi

5. **Group Aggregations:**
   - Created localized descriptive statistics (mean, std, min, max) for every numeric feature mapped against specific groupings: `gender`, `stress_level`, and `diet_type`. Example: `sleep_duration_mean_by_stress_level`.

6. **Out-of-Fold Target Encoding (OOF TE):**
   - Since Target Encoding can cause massive leakage, it was implemented strictly out-of-fold based on the unified `artifacts/folds.parquet`.
   - Mapped each categorical column across the 3 target classes using the exact cross-validation split, introducing Laplace smoothing (`smoothing = 10`) towards the global class prior.
   - For test data, the mapping applied was an average over all out-of-fold train models. No target information breaches the fold boundaries.
   - Example columns: `TE_diet_type_at-risk`, `TE_diet_type_fit`, `TE_diet_type_unhealthy`.

## Output Details
- **Train Features:** 690,088 rows, 141 columns
- **Test Features:** 295,753 rows, 141 columns

Features have been saved to `artifacts/features_train.parquet` and `artifacts/features_test.parquet`.
