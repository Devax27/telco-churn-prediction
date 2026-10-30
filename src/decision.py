"""Cost-based decision analysis for retention offers.

Model: contacting a customer costs `offer_cost`. If the customer would otherwise churn, the offer saves
them with probability `success_rate`, which is worth `value_saved`. Offers to customers who would not
churn are pure cost. All cost inputs are assumptions supplied by the caller.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.eda_utils import wilson_interval


def _validate_costs(offer_cost: float, value_saved: float, success_rate: float) -> None:
    if offer_cost <= 0 or value_saved <= 0:
        raise ValueError("offer_cost and value_saved must be positive")
    if not 0 < success_rate <= 1:
        raise ValueError("success_rate must be in (0, 1]")


def optimal_threshold(offer_cost: float, value_saved: float, success_rate: float) -> float:
    """Contact a customer when P(churn) exceeds this value, if probabilities are calibrated.

    A result of 1 or more means no customer is ever worth contacting.
    """
    _validate_costs(offer_cost, value_saved, success_rate)
    return offer_cost / (success_rate * value_saved)


def net_benefit(y_true, proba, threshold: float, offer_cost: float, value_saved: float,
                success_rate: float) -> float:
    """Total net benefit of contacting every customer whose probability is at least `threshold`."""
    _validate_costs(offer_cost, value_saved, success_rate)
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(proba, dtype=float)
    if y.shape != p.shape:
        raise ValueError("y_true and proba must have the same length")
    if np.isnan(p).any():
        raise ValueError("proba must not contain missing values")
    contacted = p >= threshold
    return float(np.sum(contacted * (success_rate * value_saved * y - offer_cost)))


def threshold_table(y_true, proba, thresholds, offer_cost: float, value_saved: float,
                    success_rate: float) -> pd.DataFrame:
    """For each threshold: share contacted, precision, recall and net benefit per 1,000 customers."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(proba, dtype=float)
    n = len(y)
    churners = float(y.sum())
    rows = []
    for t in thresholds:
        contacted = p >= t
        k = int(contacted.sum())
        caught = float(y[contacted].sum())
        rows.append(
            {
                "threshold": float(t),
                "contacted_share": k / n,
                "precision": caught / k if k else np.nan,
                "recall": caught / churners if churners else np.nan,
                "net_benefit_per_1000": net_benefit(y, p, t, offer_cost, value_saved, success_rate) / n * 1000,
            }
        )
    return pd.DataFrame(rows)


def calibration_table(y_true, proba, n_bins: int = 10) -> pd.DataFrame:
    """Group customers into equal-sized bins of predicted risk; compare predicted with observed churn."""
    df = pd.DataFrame({"y": np.asarray(y_true, dtype=float), "p": np.asarray(proba, dtype=float)})
    df["bin"] = pd.qcut(df["p"], n_bins, labels=False, duplicates="drop")
    g = df.groupby("bin").agg(customers=("y", "size"), churned=("y", "sum"), mean_predicted=("p", "mean"))
    g["observed_rate"] = g["churned"] / g["customers"]
    low, high = wilson_interval(g["churned"].to_numpy(), g["customers"].to_numpy())
    g["ci_low"] = low
    g["ci_high"] = high
    return g[["customers", "churned", "mean_predicted", "observed_rate", "ci_low", "ci_high"]]


def out_of_fold_proba(pipeline, X, y, n_splits: int = 5, n_repeats: int = 3, random_state: int = 42):
    """Predicted churn probabilities where every prediction comes from a model that did not see that
    customer. Repeats use different fold assignments and their predictions are averaged."""
    total = np.zeros(len(y))
    for r in range(n_repeats):
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state + r)
        total += cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]
    return total / n_repeats


def expected_net_benefit(probability: float, offer_cost: float, value_saved: float,
                         success_rate: float) -> float:
    """Expected net benefit of contacting ONE customer whose churn probability is `probability`.

    Positive exactly when the probability exceeds optimal_threshold(...)."""
    _validate_costs(offer_cost, value_saved, success_rate)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0 and 1")
    return success_rate * value_saved * probability - offer_cost