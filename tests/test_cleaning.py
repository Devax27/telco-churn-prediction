"""Tests for src/clean_data.py: cleaning rules on small synthetic tables, the validation rules,
and (when the raw CSV is present) an end-to-end check on the real dataset."""
import pandas as pd
import pytest

from src import clean_data

DEFAULT_ROW = {
    "customerID": "0001-AAAAA", "gender": "Female", "SeniorCitizen": "0", "Partner": "Yes",
    "Dependents": "No", "tenure": "12", "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No",
    "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check", "MonthlyCharges": "50.00",
    "TotalCharges": "600.00", "Churn": "No",
}

NO_INTERNET_ROW = {
    "InternetService": "No", "OnlineSecurity": "No internet service", "OnlineBackup": "No internet service",
    "DeviceProtection": "No internet service", "TechSupport": "No internet service",
    "StreamingTV": "No internet service", "StreamingMovies": "No internet service",
}


def raw_row(**overrides) -> dict:
    return {**DEFAULT_ROW, **overrides}


def make_raw(*rows: dict) -> pd.DataFrame:
    """Build a raw-style table with every column as text, like the real CSV is read."""
    return pd.DataFrame(list(rows), dtype=str)


# ----------------------------------------------------------------- clean()
def test_clean_renames_columns_and_converts_types():
    out = clean_data.clean(make_raw(raw_row()))
    assert out.columns.tolist() == list(clean_data.COLUMN_MAP.values()) + ["total_charges_imputed"]
    assert pd.api.types.is_integer_dtype(out["senior_citizen"])
    assert pd.api.types.is_integer_dtype(out["tenure"])
    assert pd.api.types.is_integer_dtype(out["churn"])
    assert pd.api.types.is_float_dtype(out["monthly_charges"])
    assert pd.api.types.is_float_dtype(out["total_charges"])
    assert out.loc[0, "monthly_charges"] == 50.0
    assert out.loc[0, "total_charges"] == 600.0


def test_blank_total_charges_for_tenure_zero_becomes_zero_and_is_flagged():
    raw = make_raw(
        raw_row(),
        raw_row(customerID="0002-BBBBB", tenure="0", MonthlyCharges="20.25", TotalCharges=" "),
    )
    out = clean_data.clean(raw)
    assert out["total_charges_imputed"].tolist() == [False, True]
    assert out.loc[0, "total_charges"] == 600.0
    assert out.loc[1, "total_charges"] == 0.0


def test_blank_total_charges_with_positive_tenure_raises():
    raw = make_raw(raw_row(tenure="5", TotalCharges=" "))
    with pytest.raises(ValueError, match="tenure > 0"):
        clean_data.clean(raw)


def test_churn_is_mapped_to_one_and_zero():
    out = clean_data.clean(make_raw(raw_row(Churn="Yes"), raw_row(customerID="0002-BBBBB", Churn="No")))
    assert out["churn"].tolist() == [1, 0]


def test_unexpected_churn_value_raises():
    with pytest.raises(ValueError, match="Churn"):
        clean_data.clean(make_raw(raw_row(Churn="Maybe")))


def test_missing_raw_column_raises():
    raw = make_raw(raw_row()).drop(columns=["Contract"])
    with pytest.raises(ValueError, match="missing columns"):
        clean_data.clean(raw)


def test_whitespace_is_stripped():
    out = clean_data.clean(make_raw(raw_row(Contract=" Month-to-month ", Partner="Yes ")))
    assert out.loc[0, "contract"] == "Month-to-month"
    assert out.loc[0, "partner"] == "Yes"


def test_non_numeric_tenure_raises():
    with pytest.raises(ValueError):
        clean_data.clean(make_raw(raw_row(tenure="abc")))


def test_clean_does_not_modify_its_input():
    raw = make_raw(raw_row(), raw_row(customerID="0002-BBBBB", tenure="0", TotalCharges=" "))
    before = raw.copy()
    clean_data.clean(raw)
    pd.testing.assert_frame_equal(raw, before)


def test_allowed_values_cover_every_categorical_column():
    non_categorical = {"customer_id", "senior_citizen", "tenure", "monthly_charges", "total_charges", "churn"}
    categorical = set(clean_data.COLUMN_MAP.values()) - non_categorical
    assert set(clean_data.ALLOWED_VALUES) == categorical


# -------------------------------------------------------------- validate()
@pytest.fixture
def valid_df(monkeypatch):
    """Four consistent customers, with the expected counts patched to match."""
    raw = make_raw(
        raw_row(customerID="0001-AAAAA"),
        raw_row(customerID="0002-BBBBB", tenure="0", MonthlyCharges="20.25", TotalCharges=" "),
        raw_row(customerID="0003-CCCCC", PhoneService="No", MultipleLines="No phone service", Churn="Yes"),
        raw_row(customerID="0004-DDDDD", MonthlyCharges="20.00", TotalCharges="240.00", **NO_INTERNET_ROW),
    )
    monkeypatch.setattr(clean_data, "EXPECTED_ROWS", 4)
    monkeypatch.setattr(clean_data, "EXPECTED_CHURNERS", 1)
    monkeypatch.setattr(clean_data, "EXPECTED_IMPUTED_TOTALS", 1)
    return clean_data.clean(raw)


def test_validate_accepts_consistent_data(valid_df):
    clean_data.validate(valid_df)


def test_validate_rejects_duplicate_customer_ids(valid_df):
    valid_df.loc[1, "customer_id"] = valid_df.loc[0, "customer_id"]
    with pytest.raises(ValueError, match="customer_id is unique"):
        clean_data.validate(valid_df)


def test_validate_rejects_unexpected_category(valid_df):
    valid_df.loc[0, "contract"] = "Three year"
    with pytest.raises(ValueError, match="contract contains only expected values"):
        clean_data.validate(valid_df)


def test_validate_rejects_no_phone_service_mismatch(valid_df):
    valid_df.loc[0, "multiple_lines"] = "No phone service"  # but phone_service is Yes
    with pytest.raises(ValueError, match="No phone service"):
        clean_data.validate(valid_df)


def test_validate_rejects_no_internet_service_mismatch(valid_df):
    valid_df.loc[3, "tech_support"] = "No"  # but internet_service is No
    with pytest.raises(ValueError, match="in tech_support"):
        clean_data.validate(valid_df)


def test_validate_rejects_wrong_row_count(valid_df, monkeypatch):
    monkeypatch.setattr(clean_data, "EXPECTED_ROWS", 5)
    with pytest.raises(ValueError, match="rows"):
        clean_data.validate(valid_df)


def test_validate_rejects_non_positive_monthly_charges(valid_df):
    valid_df.loc[0, "monthly_charges"] = 0.0
    with pytest.raises(ValueError, match="monthly_charges are positive"):
        clean_data.validate(valid_df)


# ----------------------------------------------------------------- check()
def test_check_raises_when_condition_is_false():
    with pytest.raises(ValueError, match="Validation failed"):
        clean_data.check(False, "something is wrong")


def test_check_passes_when_condition_is_true():
    clean_data.check(True, "everything is fine")


# ---------------------------------------------------- real data (optional)
@pytest.mark.skipif(not clean_data.RAW_PATH.exists(), reason="raw dataset not downloaded")
def test_real_dataset_cleans_and_validates():
    raw = pd.read_csv(clean_data.RAW_PATH, dtype=str, keep_default_na=False)
    df = clean_data.clean(raw)
    clean_data.validate(df)
    assert df["churn"].mean() == pytest.approx(0.2654, abs=0.0001)