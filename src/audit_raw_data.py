"""Read-only audit of the raw Telco churn CSV. Prints findings; changes nothing.

Run from the project root:  python src\\audit_raw_data.py
"""
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def find_csv() -> Path:
    files = sorted(RAW_DIR.glob("*.csv"))
    if len(files) != 1:
        raise SystemExit(f"Expected exactly one CSV in {RAW_DIR}, found {len(files)}: {[f.name for f in files]}")
    return files[0]


def to_number(series: pd.Series) -> pd.Series:
    """Convert text to numbers; blanks and non-numeric text become missing."""
    stripped = series.str.strip()
    return pd.to_numeric(stripped.where(stripped != ""), errors="coerce")


def main() -> None:
    path = find_csv()
    # Read everything as text and keep blanks as '' so nothing is silently converted or dropped.
    df = pd.read_csv(path, dtype=str, keep_default_na=False)

    section("1. FILE AND SHAPE")
    print(f"file: {path.name}")
    print(f"rows: {len(df):,} | columns: {df.shape[1]}")
    print("columns:", ", ".join(df.columns))

    section("2. COLUMN PROFILE (all values read as text)")
    profile, numeric_cols = [], []
    for col in df.columns:
        s = df[col]
        stripped = s.str.strip()
        blank = stripped == ""
        nums = to_number(s)
        numeric_like = bool((~blank).any()) and bool(nums[~blank].notna().all())
        if numeric_like:
            numeric_cols.append(col)
        profile.append({
            "column": col,
            "unique": int(s.nunique()),
            "blank": int(blank.sum()),
            "padded": int((s != stripped).sum()),
            "all_non_blank_numeric": numeric_like,
        })
    print(pd.DataFrame(profile).to_string(index=False))
    print("(blank = empty or spaces only; padded = leading or trailing spaces)")

    section("3. FIRST ROWS")
    print("not printed: the dataset's license is unclear, so this report contains only aggregate statistics")

    section("4. DUPLICATES")
    print(f"fully duplicated rows: {int(df.duplicated().sum()):,}")
    if "customerID" in df.columns:
        print(f"duplicate customerID values: {int(df['customerID'].duplicated().sum()):,}")
    else:
        print("column 'customerID' not found")

    section("5. VALUES OF LOW-CARDINALITY COLUMNS (10 or fewer distinct values)")
    for col in df.columns:
        if df[col].nunique() <= 10:
            print(f"\n{col}:")
            print(df[col].value_counts().to_string())

    section("6. NUMERIC-LOOKING COLUMNS WITH MORE THAN 10 DISTINCT VALUES")
    for col in numeric_cols:
        if df[col].nunique() > 10:
            print(f"\n{col}:")
            print(to_number(df[col]).describe().to_string())

    section("7. TARGET: Churn")
    if "Churn" in df.columns:
        counts = df["Churn"].value_counts()
        print(counts.to_string())
        print((100 * counts / counts.sum()).round(2).astype(str).add("%").to_string())
    else:
        print("column 'Churn' not found")

    section("8. TotalCharges CHECKS")
    needed = {"tenure", "MonthlyCharges", "TotalCharges"}
    if needed <= set(df.columns):
        tenure = to_number(df["tenure"])
        monthly = to_number(df["MonthlyCharges"])
        total = to_number(df["TotalCharges"])
        missing_total = total.isna()
        print(f"rows with missing or non-numeric TotalCharges: {int(missing_total.sum())}")
        if missing_total.any():
            print("tenure values among those rows:", sorted(tenure[missing_total].unique().tolist()))
            print("raw TotalCharges values among those rows:", sorted(df.loc[missing_total, "TotalCharges"].unique().tolist()))
        expected = tenure * monthly
        ok = total.notna() & (expected > 0)
        rel = (total[ok] - expected[ok]).abs() / expected[ok]
        print("\nrelative difference |TotalCharges - tenure*MonthlyCharges| / (tenure*MonthlyCharges):")
        print(rel.describe(percentiles=[0.5, 0.9, 0.99]).to_string())
    else:
        print(f"columns {sorted(needed - set(df.columns))} not found")


if __name__ == "__main__":
    main()