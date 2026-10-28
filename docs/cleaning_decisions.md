# Cleaning decisions

Source: IBM Telco Customer Churn sample dataset (Kaggle: blastchar/telco-customer-churn). The company is fictional.
Evidence: `docs/raw_data_audit.txt`. The raw file in `data/raw` is never modified; `src/clean_data.py` writes `data/processed/telco_clean.parquet`.

| Issue (evidence) | Decision |
|---|---|
| 11 rows have a blank `TotalCharges` (a single space); all 11 have `tenure` = 0 | Set to 0.0 and flagged in `total_charges_imputed`. The script raises an error if a blank ever appears for `tenure` > 0 |
| `SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges` are read as text | Converted to integer, integer, float, float |
| `Churn` is Yes/No text (1,869 Yes of 7,043 = 26.54%) | `churn` integer, 1 = left, 0 = stayed |
| "No phone service" (682) and "No internet service" (1,526) labels | Kept as recorded; the script checks they match `phone_service` and `internet_service` exactly. Handled in feature engineering |
| No duplicate rows or customer ids; no other missing values | No rows removed |
| Mixed CamelCase column names | Renamed to snake_case; `customerID` becomes `customer_id` |
| `TotalCharges` is close to `tenure` x `MonthlyCharges` (median gap 2%, 90th percentile 7.5%, max 57%) | No change; collinearity is a modelling concern |
| Target is imbalanced (26.54% positive) | No resampling at the cleaning stage; handled at modelling (stratified splits, PR-AUC, cost-based threshold) |
| Single snapshot, no dates, fictional company | Random stratified split; documented as a limitation |