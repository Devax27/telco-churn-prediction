"""Tests for src/modeling.py. Most use small synthetic data; one uses the real cleaned table if present."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from src import clean_data, modeling


def make_df(n: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Synthetic customers whose churn probability falls smoothly with tenure."""
    rng = np.random.default_rng(seed)
    data = {col: rng.choice(["x", "y", "z"], size=n) for col in modeling.CATEGORICAL_FEATURES}
    tenure = rng.integers(0, 73, size=n)
    monthly = rng.uniform(20, 110, size=n).round(2)
    data["senior_citizen"] = rng.integers(0, 2, size=n)
    data["tenure"] = tenure
    data["monthly_charges"] = monthly
    data["total_charges"] = (tenure * monthly).round(2)
    prob = 1 / (1 + np.exp((tenure - 30) / 10))
    data[modeling.TARGET] = (rng.random(n) < prob).astype(int)
    return pd.DataFrame(data)


# ------------------------------------------------------------- feature lists
def test_feature_lists_are_disjoint_and_exclude_identifiers_and_target():
    assert not set(modeling.NUMERIC_FEATURES) & set(modeling.CATEGORICAL_FEATURES)
    assert len(modeling.FEATURES) == 19
    for excluded in ["customer_id", "churn", "total_charges_imputed"]:
        assert excluded not in modeling.FEATURES


def test_features_cover_every_cleaned_column_except_the_id_and_target():
    cleaned_columns = set(clean_data.COLUMN_MAP.values())
    assert set(modeling.FEATURES) == cleaned_columns - {"customer_id", "churn"}


def test_load_clean_rejects_a_table_with_missing_columns(tmp_path):
    path = tmp_path / "bad.parquet"
    make_df(50).drop(columns=["contract"]).to_parquet(path)
    with pytest.raises(ValueError, match="missing columns"):
        modeling.load_clean(path)


# --------------------------------------------------------------------- split
def test_split_sizes_stratification_and_disjointness():
    df = make_df(1000)
    X_train, X_test, y_train, y_test = modeling.split_data(df)
    assert len(X_train) == 800 and len(X_test) == 200
    assert set(X_train.index).isdisjoint(X_test.index)
    assert set(X_train.index) | set(X_test.index) == set(df.index)
    overall = df[modeling.TARGET].mean()
    assert y_train.mean() == pytest.approx(overall, abs=0.01)
    assert y_test.mean() == pytest.approx(overall, abs=0.01)
    assert list(X_train.columns) == modeling.FEATURES


def test_split_is_reproducible_and_depends_on_the_seed():
    df = make_df(1000)
    first = modeling.split_data(df)[1].index
    again = modeling.split_data(df)[1].index
    other = modeling.split_data(df, random_state=7)[1].index
    assert first.equals(again)
    assert not first.equals(other)


# ------------------------------------------------------------------ pipelines
def test_logistic_pipeline_returns_valid_probabilities():
    df = make_df(500)
    X, y = df[modeling.FEATURES], df[modeling.TARGET]
    proba = modeling.build_logistic_pipeline().fit(X, y).predict_proba(X)
    assert proba.shape == (500, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_pipeline_handles_a_category_it_has_never_seen():
    df = make_df(500)
    X, y = df[modeling.FEATURES], df[modeling.TARGET]
    model = modeling.build_logistic_pipeline().fit(X, y)
    new = X.iloc[:3].copy()
    new["contract"] = "a value never seen in training"
    proba = model.predict_proba(new)
    assert proba.shape == (3, 2)


def test_baseline_predicts_the_training_churn_rate_for_everyone():
    df = make_df(500)
    X, y = df[modeling.FEATURES], df[modeling.TARGET]
    proba = modeling.build_baseline_pipeline().fit(X, y).predict_proba(X)[:, 1]
    assert np.allclose(proba, y.mean())


def test_logistic_pipeline_beats_the_base_rate_on_data_with_a_clear_signal():
    df = make_df(2000)
    X_train, X_test, y_train, y_test = modeling.split_data(df)
    model = modeling.build_logistic_pipeline().fit(X_train, y_train)
    ap = average_precision_score(y_test, model.predict_proba(X_test)[:, 1])
    assert ap > y_test.mean() + 0.1


# ----------------------------------------------------------- real data (optional)
@pytest.mark.skipif(not modeling.DATA_PATH.exists(), reason="cleaned dataset not generated")
def test_real_dataset_split_has_the_expected_sizes_and_churn_counts():
    df = modeling.load_clean()
    X_train, X_test, y_train, y_test = modeling.split_data(df)
    assert len(X_train) == 5_634 and len(X_test) == 1_409
    assert int(y_train.sum()) + int(y_test.sum()) == 1_869
    assert abs(int(y_test.sum()) - 374) <= 1