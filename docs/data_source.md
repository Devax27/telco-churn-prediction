# Data source and licensing

**Dataset:** Telco Customer Churn, an IBM sample dataset about a fictional telecommunications company, downloaded from Kaggle (`blastchar/telco-customer-churn`).

**License as shown on Kaggle (checked 2026-10-28):** "Data files © Original Authors". This is not an open license and does not grant redistribution rights.

**A third-party claim not relied on here:** a Hugging Face mirror of the dataset lists CC BY 4.0. That is a statement about a copy, not the original source.

## How this repository handles it

- The raw CSV and all processed data files are **not committed** (`data/raw/*` and `data/processed/*` are git-ignored). The README tells readers to download the file from Kaggle themselves.
- Documents and the README contain only aggregate statistics (counts, rates, charts), not rows of customer data.
- The company and customers are fictional, so results illustrate a method and say nothing about real customers.

## Open decision

Whether to publish a model trained on this data (needed for a public demo app). To be decided before deployment, after reading IBM's terms for the sample data.