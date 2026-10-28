"""Clean the raw Telco churn CSV and write an analysis-ready Parquet file.

Rules and evidence: docs/cleaning_decisions.md and docs/raw_data_audit.txt.
The raw file is never modified.

Run from the project root:  python src\\clean_data.py
"""
import logging
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUT_PATH = ROOT / "data" / "processed" / "telco_clean.parquet"

COLUMN_MAP = {
    "customerID": "customer_id",
    "gender": "gender",
    "SeniorCitizen": "senior_citizen",
    "Partner": "partner",
    "Dependents": "dependents",
    "tenure": "tenure",
    "PhoneService": "phone_service",
    "MultipleLines": "multiple_lines",
    "InternetService": "internet_service",
    "OnlineSecurity": "online_security",
    "OnlineBackup": "online_backup",
    "DeviceProtection": "device_protection",
    "TechSupport": "tech_support",
    "StreamingTV": "streaming_tv",
    "StreamingMovies": "streaming_movies",
    "Contract": "contract",
    "PaperlessBilling": "paperless_billing",
    "PaymentMethod": "payment_method",
    "MonthlyCharges": "monthly_charges",
    "TotalCharges": "total_charges",
    "Churn": "churn",
}

INTERNET_ADDON_COLUMNS = [
    "online_security", "online_backup", "device_protection",
    "tech_support", "streaming_tv", "streaming_movies",
]

YES_NO = {"Yes", "No"}
ALLOWED_VALUES = {
    "gender": {"Female", "Male"},
    "partner": YES_NO,
    "dependents": YES_NO,
    "phone_service": YES_NO,
    "multiple_lines": {"Yes", "No", "No phone service"},
    "internet_service": {"DSL", "Fiber optic", "No"},
    "contract": {"Month-to-month", "One year", "Two year"},
    "paperless_billing": YES_NO,
    "payment_method": {
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    },
    **{col: {"Yes", "No", "No internet service"} for col in INTERNET_ADDON_COLUMNS},
}

# Counts observed in docs/raw_data_audit.txt.
EXPECTED_ROWS = 7_043
EXPECTED_CHURNERS = 1_869
EXPECTED_IMPUTED_TOTALS = 11

logger = logging.getLogger("clean_data")


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the documented cleaning rules to the raw table (all columns read as text)."""
    missing = set(COLUMN_MAP) - set(raw.columns)
    if missing:
        raise ValueError(f"Raw data is missing columns: {sorted(missing)}")

    df = raw[list(COLUMN_MAP)].rename(columns=COLUMN_MAP).copy()
    for col in df.columns:
        df[col] = df[col].str.strip()

    df["senior_citizen"] = pd.to_numeric(df["senior_citizen"], errors="raise").astype(int)
    df["tenure"] = pd.to_numeric(df["tenure"], errors="raise").astype(int)
    df["monthly_charges"] = pd.to_numeric(df["monthly_charges"], errors="raise").astype(float)

    # Blank TotalCharges -> missing, then 0.0 for customers with tenure 0 (see cleaning_decisions.md).
    total = df["total_charges"].where(df["total_charges"] != "")
    df["total_charges"] = pd.to_numeric(total, errors="raise").astype(float)
    df["total_charges_imputed"] = df["total_charges"].isna()
    unexplained = df["total_charges_imputed"] & (df["tenure"] != 0)
    if unexplained.any():
        raise ValueError(
            f"{int(unexplained.sum())} rows have a blank TotalCharges but tenure > 0; "
            "the tenure-0 rule does not explain them"
        )
    df.loc[df["total_charges_imputed"], "total_charges"] = 0.0

    df["churn"] = df["churn"].map({"Yes": 1, "No": 0})
    if df["churn"].isna().any():
        raise ValueError("Unexpected values in the Churn column")
    df["churn"] = df["churn"].astype(int)
    return df


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"Validation failed: {message}")
    logger.info("OK: %s", message)


def validate(df: pd.DataFrame) -> None:
    check(len(df) == EXPECTED_ROWS, f"{EXPECTED_ROWS:,} rows")
    check(df["customer_id"].is_unique, "customer_id is unique")
    check(int(df.isna().sum().sum()) == 0, "no missing values remain")

    for col, allowed in ALLOWED_VALUES.items():
        check(set(df[col].unique()) <= allowed, f"{col} contains only expected values")
    check(set(df["senior_citizen"].unique()) <= {0, 1}, "senior_citizen is 0 or 1")
    check(df["tenure"].between(0, 72).all(), "tenure is between 0 and 72")
    check((df["monthly_charges"] > 0).all(), "monthly_charges are positive")
    check((df["total_charges"] >= 0).all(), "total_charges are not negative")

    check(int(df["churn"].sum()) == EXPECTED_CHURNERS, f"{EXPECTED_CHURNERS:,} churned customers")
    imputed = df[df["total_charges_imputed"]]
    check(len(imputed) == EXPECTED_IMPUTED_TOTALS, f"{EXPECTED_IMPUTED_TOTALS} imputed TotalCharges rows")
    check((imputed["tenure"] == 0).all(), "every imputed row has tenure 0")

    no_phone = df["phone_service"] == "No"
    check(
        bool((no_phone == (df["multiple_lines"] == "No phone service")).all()),
        "'No phone service' matches phone_service = No exactly",
    )
    no_internet = df["internet_service"] == "No"
    for col in INTERNET_ADDON_COLUMNS:
        check(
            bool((no_internet == (df[col] == "No internet service")).all()),
            f"'No internet service' in {col} matches internet_service = No exactly",
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not RAW_PATH.exists():
        raise SystemExit(f"Raw file not found: {RAW_PATH}")

    # Read everything as text; keep blanks as '' so nothing is silently converted.
    raw = pd.read_csv(RAW_PATH, dtype=str, keep_default_na=False)
    df = clean(raw)

    imputed = df[df["total_charges_imputed"]]
    logger.info(
        "TotalCharges set to 0.0 for %d tenure-0 customers; their churn labels: %s",
        len(imputed), imputed["churn"].value_counts().to_dict(),
    )
    logger.info("churn rate: %.2f%%", 100 * df["churn"].mean())

    validate(df)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    logger.info("wrote %s (%s rows, %d columns)", OUT_PATH.name, f"{len(df):,}", df.shape[1])
    print()
    print(df.dtypes.to_string())


if __name__ == "__main__":
    main()