"""Tests for src/final_evaluation.py. Expected values were calculated by hand."""
import sys

import numpy as np
import pytest

from src import final_evaluation as fe

# Four customers: churn = 1, 0, 1, 0 with predicted probabilities 0.9, 0.8, 0.3, 0.1.
Y = np.array([1, 0, 1, 0])
P = np.array([0.9, 0.8, 0.3, 0.1])
COSTS = dict(offer_cost=10.0, value_saved=100.0, success_rate=0.5)


def synthetic(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0, 1, n)
    y = (rng.random(n) < p).astype(int)
    return y, p


def test_point_metrics_hand_calculated():
    m = fe.point_metrics(Y, P, 0.5, COSTS)
    assert m["pr_auc"] == pytest.approx((1 + 2 / 3) / 2)       # precision at the two positives: 1/1 and 2/3
    assert m["roc_auc"] == pytest.approx(0.75)                  # 3 of 4 positive-negative pairs ranked correctly
    assert m["brier"] == pytest.approx(0.2875)                  # (0.01 + 0.64 + 0.49 + 0.01) / 4
    assert m["contacted_share"] == pytest.approx(0.5)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["net_benefit_per_1000"] == pytest.approx(7500.0)   # total 30 / 4 customers x 1000
    assert m["net_benefit_contact_all_per_1000"] == pytest.approx(15000.0)   # total 60 / 4 x 1000
    assert m["gain_vs_contact_all_per_1000"] == pytest.approx(-7500.0)


def test_bootstrap_is_deterministic_for_a_seed_and_differs_across_seeds():
    y, p = synthetic()
    a = fe.bootstrap_metrics(y, p, 0.3, COSTS, n_boot=50, seed=1)
    b = fe.bootstrap_metrics(y, p, 0.3, COSTS, n_boot=50, seed=1)
    c = fe.bootstrap_metrics(y, p, 0.3, COSTS, n_boot=50, seed=2)
    assert a.equals(b)
    assert not a.equals(c)


def test_bootstrap_interval_contains_the_estimate():
    y, p = synthetic()
    draws = fe.bootstrap_metrics(y, p, 0.3, COSTS, n_boot=300, seed=3)
    table = fe.summarise(fe.point_metrics(y, p, 0.3, COSTS), draws).set_index("metric")
    for metric in ["pr_auc", "roc_auc", "brier"]:
        row = table.loc[metric]
        assert row["ci_low"] <= row["estimate"] <= row["ci_high"]
        assert row["ci_low"] < row["ci_high"]


def test_bootstrap_skips_single_class_resamples():
    y = np.array([0, 0, 0, 1])
    p = np.array([0.1, 0.2, 0.3, 0.9])
    draws = fe.bootstrap_metrics(y, p, 0.5, COSTS, n_boot=200, seed=0)
    assert 0 < len(draws) < 200


def test_perfect_predictions_give_a_degenerate_interval():
    y = np.array([0, 1] * 50)
    draws = fe.bootstrap_metrics(y, y.astype(float), 0.5, COSTS, n_boot=100, seed=0)
    assert draws["pr_auc"].quantile(0.025) == pytest.approx(1.0)
    assert draws["pr_auc"].quantile(0.975) == pytest.approx(1.0)


def test_script_refuses_to_run_twice_without_the_rerun_flag(tmp_path, monkeypatch):
    results = tmp_path / "final_test_results.csv"
    results.write_text("already here")
    monkeypatch.setattr(fe, "OUT_PATH", results)
    monkeypatch.setattr(sys, "argv", ["final_evaluation.py"])
    with pytest.raises(SystemExit) as exc:
        fe.main()
    assert "already exists" in str(exc.value)