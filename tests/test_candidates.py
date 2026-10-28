"""Tests for the candidate models and the paired comparison helper."""
import numpy as np
import pytest
from scipy import sparse

from src import modeling, train_compare
from tests.test_modeling import make_df

NAMES = list(modeling.candidate_models())


def test_candidates_include_the_reference_and_have_unique_names():
    assert train_compare.REFERENCE in NAMES
    assert len(NAMES) == 5
    assert len(set(NAMES)) == 5


@pytest.mark.parametrize("name", NAMES)
def test_every_candidate_fits_and_returns_valid_probabilities(name):
    df = make_df(600)
    X, y = df[modeling.FEATURES], df[modeling.TARGET]
    model = modeling.candidate_models()[name].fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (600, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_tree_candidates_handle_an_unseen_category():
    df = make_df(600)
    X, y = df[modeling.FEATURES], df[modeling.TARGET]
    new = X.iloc[:3].copy()
    new["contract"] = "a value never seen in training"
    for name in ["random forest", "gradient boosting"]:
        model = modeling.candidate_models()[name].fit(X, y)
        assert model.predict_proba(new).shape == (3, 2)


def test_tree_preprocessor_output_is_dense():
    df = make_df(300)
    out = modeling.build_tree_preprocessor().fit_transform(df[modeling.FEATURES])
    assert not sparse.issparse(out)
    assert out.shape[0] == 300


def test_paired_summary_values():
    s = train_compare.paired_summary([0.7, 0.6, 0.5], [0.6, 0.6, 0.6])
    assert s["mean_diff"] == pytest.approx(0.0, abs=1e-12)
    assert s["std_diff"] == pytest.approx(0.1)
    assert s["win_rate"] == pytest.approx(1 / 3)


def test_paired_summary_rejects_different_fold_counts():
    with pytest.raises(ValueError):
        train_compare.paired_summary([0.7, 0.6], [0.6, 0.6, 0.6])