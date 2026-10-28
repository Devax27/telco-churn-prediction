"""Tests for src/eda_utils.py. Expected intervals were calculated by hand from the Wilson formula."""
import numpy as np
import pandas as pd
import pytest

from src.eda_utils import rate_table, wilson_interval


def test_wilson_interval_for_five_of_ten():
    low, high = wilson_interval(5, 10)
    assert low == pytest.approx(0.2366, abs=0.001)
    assert high == pytest.approx(0.7634, abs=0.001)


def test_wilson_interval_for_zero_successes():
    low, high = wilson_interval(0, 10)
    assert low == pytest.approx(0.0, abs=1e-12)
    assert high == pytest.approx(0.2775, abs=0.001)


def test_wilson_interval_contains_the_estimate_and_narrows_with_more_data():
    low_small, high_small = wilson_interval(30, 100)
    low_large, high_large = wilson_interval(300, 1000)
    assert low_small < 0.3 < high_small
    assert (high_large - low_large) < (high_small - low_small)


def test_wilson_interval_is_vectorised():
    low, high = wilson_interval(np.array([1, 5]), np.array([10, 10]))
    assert low.shape == (2,)
    assert high.shape == (2,)
    assert low[0] < low[1]


def test_wilson_interval_rejects_invalid_input():
    with pytest.raises(ValueError):
        wilson_interval(1, 0)
    with pytest.raises(ValueError):
        wilson_interval(11, 10)


def test_rate_table_counts_rates_and_intervals():
    df = pd.DataFrame({"g": ["a", "a", "a", "b", "b"], "churn": [1, 0, 1, 0, 0]})
    t = rate_table(df, "g")
    assert t.loc["a", "customers"] == 3
    assert t.loc["a", "churned"] == 2
    assert t.loc["a", "churn_rate"] == pytest.approx(2 / 3)
    assert t.loc["b", "churned"] == 0
    assert t.loc["b", "churn_rate"] == 0
    assert (t["ci_low"] <= t["churn_rate"]).all()
    assert (t["churn_rate"] <= t["ci_high"]).all()
    
@pytest.mark.parametrize("k, n", [(0, 11), (11, 11), (0, 1), (1, 1), (5, 10), (784, 1470)])
def test_wilson_interval_always_contains_the_estimate(k, n):
    low, high = wilson_interval(k, n)
    assert low <= k / n <= high