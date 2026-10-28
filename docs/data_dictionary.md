# Data dictionary

Cleaned table: `data/processed/telco_clean.parquet` (7,043 rows, 22 columns, one row per customer, a single snapshot with no dates).
Counts come from `docs/raw_data_audit.txt`. The source does not publish units for every column; where I inferred one, it says so.

| Column | Type | Meaning | Values or range |
|---|---|---|---|
| customer_id | text | Unique customer identifier | 7,043 distinct values |
| gender | text | Customer gender as recorded | Female (3,488), Male (3,555) |
| senior_citizen | integer | Senior-citizen flag (the age rule is not documented; 1 presumably means senior) | 0 (5,901), 1 (1,142) |
| partner | text | Has a partner | Yes (3,402), No (3,641) |
| dependents | text | Has dependents | Yes (2,110), No (4,933) |
| tenure | integer | Time as a customer. Unit inferred as months, because total_charges is close to tenure x monthly_charges | 0 to 72 |
| phone_service | text | Has phone service | Yes (6,361), No (682) |
| multiple_lines | text | Has multiple phone lines | Yes (2,971), No (3,390), No phone service (682) |
| internet_service | text | Type of internet service | Fiber optic (3,096), DSL (2,421), No (1,526) |
| online_security | text | Has the online security add-on | Yes (2,019), No (3,498), No internet service (1,526) |
| online_backup | text | Has the online backup add-on | Yes (2,429), No (3,088), No internet service (1,526) |
| device_protection | text | Has the device protection add-on | Yes (2,422), No (3,095), No internet service (1,526) |
| tech_support | text | Has the tech support add-on | Yes (2,044), No (3,473), No internet service (1,526) |
| streaming_tv | text | Streams TV | Yes (2,707), No (2,810), No internet service (1,526) |
| streaming_movies | text | Streams movies | Yes (2,732), No (2,785), No internet service (1,526) |
| contract | text | Contract type | Month-to-month (3,875), One year (1,473), Two year (1,695) |
| paperless_billing | text | Uses paperless billing | Yes (4,171), No (2,872) |
| payment_method | text | Payment method | Electronic check (2,365), Mailed check (1,612), Bank transfer (automatic) (1,544), Credit card (automatic) (1,522) |
| monthly_charges | float | Monthly charge (currency not documented) | 18.25 to 118.75 |
| total_charges | float | Total billed to date (inferred). Set to 0.0 for the 11 customers with tenure 0 | 0.0 to 8,684.80 |
| churn | integer | **Target.** 1 = left within the last month (per the source description), 0 = stayed | 1 (1,869 = 26.54%), 0 (5,174) |
| total_charges_imputed | boolean | True for the 11 rows whose total_charges was blank in the raw file | 11 True |

## Known quirks

- "No phone service" and "No internet service" repeat information from `phone_service` and `internet_service`; `src/clean_data.py` verifies they match exactly.
- `tenure` of churned customers is how long they stayed; for active customers it is how long they have stayed so far. A model can partly learn "short tenure means left", which may look better than it would when scoring live customers.
- Single snapshot: there is no time dimension, so a time-based split is impossible.