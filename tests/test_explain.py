"""Tests for src/explain.py, on synthetic data where only tenure-related inputs matter."""
import numpy as np
import pandas as pd
import pytest

from src import explain, modeling
from tests.test_modeling import make_df


def make_xy(n: int = 3000, seed: int = 1):
    df = make_df(n, seed=seed)
    return df[modeling.FEATURES], df[modeling.TARGET]


def test_feature_groups_cover_every_feature_and_joint_columns_exist():
    groups = explain.feature_groups()
    assert len(groups) == len(modeling.FEATURES) + len(explain.JOINT_GROUPS)
    for feature in modeling.FEATURES:
        assert groups[feature] == [feature]
    for columns in explain.JOINT_GROUPS.values():
        assert set(columns) <= set(modeling.FEATURES)


def test_permute_columns_jointly_keeps_the_shuffled_columns_aligned():
    X = pd.DataFrame({"a": np.arange(50), "b": np.arange(50) * 10, "c": np.arange(50) + 100})
    before = X.copy()
    out = explain.permute_columns_jointly(X, ["a", "b"], np.random.default_rng(0))
    assert (out["b"] == out["a"] * 10).all()          # rows moved together
    assert not (out["a"] == X["a"]).all()              # and they did move
    assert sorted(out["a"]) == sorted(X["a"])          # same values, new order
    assert (out["c"] == X["c"]).all()                  # other columns untouched
    pd.testing.assert_frame_equal(X, before)           # the input was not modified


def test_permute_columns_jointly_works_with_a_non_default_index():
    X = pd.DataFrame({"a": np.arange(30), "b": np.arange(30) * 2}, index=np.arange(100, 130))
    out = explain.permute_columns_jointly(X, ["a"], np.random.default_rng(3))
    assert out.index.equals(X.index)
    assert sorted(out["a"]) == sorted(X["a"])


def test_permute_columns_jointly_rejects_unknown_columns():
    X = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError, match="unknown columns"):
        explain.permute_columns_jointly(X, ["nope"], np.random.default_rng(0))


def test_permutation_drop_is_large_for_the_signal_and_small_for_noise():
    X, y = make_xy()
    model = modeling.build_logistic_pipeline().fit(X, y)
    rng = np.random.default_rng(0)
    signal = explain.permutation_drop(model, X, y, ["tenure", "monthly_charges", "total_charges"], 5, rng)
    noise = explain.permutation_drop(model, X, y, ["gender"], 5, rng)
    assert signal > 0.05
    assert abs(noise) < 0.03
    assert noise < signal


def test_cv_permutation_importance_ranks_the_signal_first():
    X, y = make_xy(2000)
    groups = {"tenure group": ["tenure", "monthly_charges", "total_charges"], "gender": ["gender"], "contract": ["contract"]}
    out = explain.cv_permutation_importance(
        modeling.build_logistic_pipeline, X, y, groups, n_splits=3, n_repeats=3, random_state=0
    )
    assert set(out["group"]) == set(groups)
    assert out.loc[0, "group"] == "tenure group"
    assert out["mean_drop"].is_monotonic_decreasing
    assert (out["std_across_folds"] >= 0).all()
    assert (out["min_fold"] <= out["mean_drop"]).all() and (out["mean_drop"] <= out["max_fold"]).all()


def test_cv_permutation_importance_is_deterministic():
    X, y = make_xy(1200)
    groups = {"tenure": ["tenure"], "gender": ["gender"]}
    kwargs = dict(n_splits=3, n_repeats=2, random_state=5)
    first = explain.cv_permutation_importance(modeling.build_logistic_pipeline, X, y, groups, **kwargs)
    second = explain.cv_permutation_importance(modeling.build_logistic_pipeline, X, y, groups, **kwargs)
    pd.testing.assert_frame_equal(first, second)


def test_coefficient_table_structure_and_direction():
    X, y = make_xy(1500)
    model = modeling.build_logistic_pipeline().fit(X, y)
    table = explain.coefficient_table(model)
    assert len(table) == len(model.named_steps["model"].coef_[0])
    assert table["feature"].str.startswith(("num__", "cat__")).all()
    assert np.allclose(table["odds_ratio"], np.exp(table["coefficient"]))
    assert table["coefficient"].abs().is_monotonic_decreasing
    coef = table.set_index("feature")["coefficient"]
    assert coef["num__tenure"] + coef["num__total_charges"] < 0   # churn falls as tenure grows in this data