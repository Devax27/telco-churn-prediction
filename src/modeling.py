"""Shared modelling code: feature lists, the train/test split, and scikit-learn pipelines."""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "telco_clean.parquet"

TARGET = "churn"
NUMERIC_FEATURES = ["senior_citizen", "tenure", "monthly_charges", "total_charges"]
CATEGORICAL_FEATURES = [
    "gender", "partner", "dependents", "phone_service", "multiple_lines", "internet_service",
    "online_security", "online_backup", "device_protection", "tech_support",
    "streaming_tv", "streaming_movies", "contract", "paperless_billing", "payment_method",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
# Deliberately NOT features: customer_id (an identifier) and total_charges_imputed
# (a data-cleaning flag that only repeats tenure == 0).

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_clean(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the cleaned table and check that every column the models need is present."""
    df = pd.read_parquet(path)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Cleaned data is missing columns: {missing}")
    return df


def split_data(df: pd.DataFrame, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):
    """Stratified train/test split. Returns X_train, X_test, y_train, y_test."""
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def build_preprocessor() -> ColumnTransformer:
    """Scale numeric columns and one-hot encode categorical ones. Unseen categories are ignored."""
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def build_baseline_pipeline() -> Pipeline:
    """Predicts the training churn rate for every customer. The floor any real model must beat."""
    return Pipeline([("prep", build_preprocessor()), ("model", DummyClassifier(strategy="prior"))])


def build_logistic_pipeline(C: float = 1.0) -> Pipeline:
    """Logistic regression with L2 regularisation (it also handles the tenure/charges collinearity)."""
    return Pipeline(
        [("prep", build_preprocessor()), ("model", LogisticRegression(C=C, max_iter=1000))]
    )