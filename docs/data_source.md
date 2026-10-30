# Data source and licensing

**Dataset:** Telco Customer Churn, an IBM sample dataset about a fictional telecommunications company, downloaded from Kaggle (`blastchar/telco-customer-churn`).

**License as shown on Kaggle (checked 2026-10-28):** "Data files © Original Authors". This is not an open license and does not grant redistribution rights.

**A third-party claim not relied on here:** a Hugging Face mirror of the dataset lists CC BY 4.0. That is a statement about a copy, not the original source.

## How this repository handles it

- The raw CSV and all processed data files are **not committed** (`data/raw/*` and `data/processed/*` are git-ignored). The README tells readers to download the file from Kaggle themselves.
- Documents and the README contain only aggregate statistics (counts, rates, charts), not rows of customer data.
- The company and customers are fictional, so results illustrate a method and say nothing about real customers.
- The saved model files in models/ are not committed either, for the same reason.


## Decision on publishing the model (2026-10-30)

Forum posts report that IBM's own page for the dataset states no license, and none of the threads I found
contains an authoritative answer. A model trained on the data is a derived artifact of unclear status, so
the saved model is **not** committed and there is **no public hosted demo**. The demo runs locally
(`streamlit run app.py`) after the reader downloads the dataset and runs `python src\train_final.py`.
The README shows screenshots of the local demo. This can be revisited if the licensing is clarified.