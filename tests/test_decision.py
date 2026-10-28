"""Tests for src/decision.py. Expected values were calculated by hand."""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src import decision, modeling
from tests.test_modeling import make_df

# Four customers: churn = 1, 0, 1, 0 with predicted probabilities 0.9, 0.8, 0.3, 0.1.
Y = np.array([1, 0, 1, 0])
P = np.array([0.9, 0.8, 0.3, 0.1])
COSTS = dict(offer_cost=10.0, value_saved=100.0, success_rate=0.5)  # one saved churner is worth 0.5 * 100 = 50


def test_optimal_threshold_is_cost_over_expected_value():
    assert decision.optimal_threshold(**COSTS) == pytest.approx(0.2)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(offer_cost=0, value_saved=100, success_rate=0.5),
        dict(offer_cost=10, value_saved=-1, success_rate=0.5),
        dict(offer_cost=10, value_saved=100, success_rate=0),
        dict(offer_cost=10, value_saved=100, success_rate=1.5),
    ],
)
def test_invalid_cost_inputs_raise(kwargs):
    with pytest.raises(ValueError):
        decision.optimal_threshold(**kwargs)


@pytest.mark.parametrize("threshold, expected", [(0.95, 0.0), (0.5, 30.0), (0.25, 70.0), (0.0, 60.0)])
def test_net_benefit_hand_calculated(threshold, expected):
    # threshold 0.5 contacts customers 1 and 2: (50 - 10) + (0 - 10) = 30
    # threshold 0.25 contacts customers 1, 2, 3: 30 + (50 - 10) = 70
    # threshold 0.0 contacts everyone: 70 + (0 - 10) = 60
    assert decision.net_benefit(Y, P, threshold, **COSTS) == pytest.approx(expected)


def test_net_benefit_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        decision.net_benefit([1, 0], [0.5], 0.5, **COSTS)


def test_net_benefit_rejects_missing_probabilities():
    with pytest.raises(ValueError, match="missing"):
        decision.net_benefit(Y, [0.9, np.nan, 0.3, 0.1], 0.5, **COSTS)


def test_threshold_table_hand_calculated():
    t = decision.threshold_table(Y, P, [0.0, 0.5, 0.95], **COSTS).set_index("threshold")
    assert t.loc[0.0, "contacted_share"] == pytest.approx(1.0)
    assert t.loc[0.0, "precision"] == pytest.approx(0.5)
    assert t.loc[0.0, "recall"] == pytest.approx(1.0)
    assert t.loc[0.0, "net_benefit_per_1000"] == pytest.approx(15000.0)
    assert t.loc[0.5, "contacted_share"] == pytest.approx(0.5)
    assert t.loc[0.5, "precision"] == pytest.approx(0.5)
    assert t.loc[0.5, "recall"] == pytest.approx(0.5)
    assert t.loc[0.5, "net_benefit_per_1000"] == pytest.approx(7500.0)
    assert t.loc[0.95, "contacted_share"] == 0
    assert np.isnan(t.loc[0.95, "precision"])
    assert t.loc[0.95, "recall"] == 0
    assert t.loc[0.95, "net_benefit_per_1000"] == 0


def test_best_threshold_on_expected_values_matches_the_theoretical_rule():
    p = np.linspace(0.005, 0.995, 100)  # 0.005, 0.015, ..., 0.995
    y = p  # soft labels: each customer churns with probability p, so benefits are expected values
    costs = dict(offer_cost=20.0, value_saved=400.0, success_rate=0.5)  # theory: 20 / 200 = 0.1
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    table = decision.threshold_table(y, p, grid, **costs)
    best = table.loc[table["net_benefit_per_1000"].idxmax(), "threshold"]
    assert decision.optimal_threshold(**costs) == pytest.approx(0.1)
    assert best == pytest.approx(0.1)


def test_calibration_table_hand_calculated():
    t = decision.calibration_table([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], n_bins=2)
    assert t["customers"].tolist() == [2, 2]
    assert t["mean_predicted"].tolist() == pytest.approx([0.15, 0.85])
    assert t["observed_rate"].tolist() == pytest.approx([0.0, 1.0])


def test_calibration_intervals_contain_the_observed_rate():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 2000)
    y = (rng.random(2000) < p).astype(int)
    t = decision.calibration_table(y, p, n_bins=10)
    assert len(t) == 10
    assert t["customers"].sum() == 2000
    assert (t["ci_low"] <= t["observed_rate"]).all()
    assert (t["observed_rate"] <= t["ci_high"]).all()
    assert (t["observed_rate"] - t["mean_predicted"]).abs().max() < 0.15


def test_out_of_fold_proba_is_valid_and_informative():
    df = make_df(600)
    X, y = df[modeling.FEATURES], df[modeling.TARGET]
    p = decision.out_of_fold_proba(modeling.build_logistic_pipeline(), X, y, n_repeats=2)
    assert p.shape == (600,)
    assert ((p >= 0) & (p <= 1)).all()
    assert roc_auc_score(y, p) > 0.65


def test_out_of_fold_proba_is_deterministic():
    df = make_df(400)
    X, y = df[modeling.FEATURES], df[modeling.TARGET]
    first = decision.out_of_fold_proba(modeling.build_logistic_pipeline(), X, y, n_repeats=2)
    second = decision.out_of_fold_proba(modeling.build_logistic_pipeline(), X, y, n_repeats=2)
    assert np.allclose(first, second)