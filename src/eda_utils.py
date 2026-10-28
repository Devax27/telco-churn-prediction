"""Small statistics helpers shared by the EDA notebook and later modelling code."""
import numpy as np
import pandas as pd


def wilson_interval(successes, trials, z: float = 1.96):
    """Wilson score interval for a proportion (95% by default).

    Works on scalars or arrays. Returns (lower, upper), each clipped to [0, 1].
    """
    k = np.asarray(successes, dtype=float)
    n = np.asarray(trials, dtype=float)
    if np.any(n <= 0):
        raise ValueError("trials must be positive")
    if np.any(k < 0) or np.any(k > n):
        raise ValueError("successes must be between 0 and trials")

    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    # Mathematically the interval always contains p, but floating-point rounding can put a bound a hair
    # on the wrong side (for example a lower bound of 1e-17 when p = 0), so enforce it explicitly.
    lower = np.minimum(np.clip(center - half, 0.0, 1.0), p)
    upper = np.maximum(np.clip(center + half, 0.0, 1.0), p)
    return lower, upper


def rate_table(df: pd.DataFrame, by, target: str = "churn") -> pd.DataFrame:
    """Customers, churned count, churn rate and a 95% Wilson interval for each group in `by`."""
    grouped = df.groupby(by, observed=True)[target].agg(customers="size", churned="sum")
    grouped["churn_rate"] = grouped["churned"] / grouped["customers"]
    low, high = wilson_interval(grouped["churned"].to_numpy(), grouped["customers"].to_numpy())
    grouped["ci_low"] = low
    grouped["ci_high"] = high
    return grouped