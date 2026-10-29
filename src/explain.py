"""Explaining the churn model: which inputs it relies on. Uses TRAINING data only.

Permutation importance: shuffle one input (or a group of related inputs together) on validation folds,
and measure how much PR-AUC drops. Related inputs are shuffled jointly because correlated columns
hide each other when shuffled one at a time.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from src import modeling

JOINT_GROUPS = {
    "[joint] tenure + monthly + total charges": ["tenure", "monthly_charges", "total_charges"],
    "[joint] internet service + 6 add-ons": [
        "internet_service", "online_security", "online_backup", "device_protection",
        "tech_support", "streaming_tv", "streaming_movies",
    ],
    "[joint] phone service + multiple lines": ["phone_service", "multiple_lines"],
    "[joint] demographics": ["gender", "senior_citizen", "partner", "dependents"],
    "[joint] paperless billing + payment method": ["paperless_billing", "payment_method"],
}


def feature_groups() -> dict:
    """Every single feature, plus the joint groups of related features."""
    groups = {col: [col] for col in modeling.FEATURES}
    groups.update(JOINT_GROUPS)
    return groups


def permute_columns_jointly(X: pd.DataFrame, columns, rng) -> pd.DataFrame:
    """Return a copy of X in which `columns` are shuffled together (one shared row permutation),
    so the relationships among those columns are preserved while their link to the rest is broken."""
    missing = [c for c in columns if c not in X.columns]
    if missing:
        raise ValueError(f"unknown columns: {missing}")
    perm = rng.permutation(len(X))
    out = X.copy()
    for col in columns:
        out[col] = X[col].to_numpy()[perm]  # positional assignment, so a non-default index is fine
    return out


def permutation_drop(model, X_val, y_val, columns, n_repeats: int, rng) -> float:
    """Mean drop in PR-AUC when `columns` are shuffled jointly, over n_repeats shuffles."""
    baseline = average_precision_score(y_val, model.predict_proba(X_val)[:, 1])
    drops = []
    for _ in range(n_repeats):
        shuffled = permute_columns_jointly(X_val, columns, rng)
        drops.append(baseline - average_precision_score(y_val, model.predict_proba(shuffled)[:, 1]))
    return float(np.mean(drops))


def cv_permutation_importance(make_pipeline, X, y, groups: dict, n_splits: int = 5, n_repeats: int = 10,
                              random_state: int = 42) -> pd.DataFrame:
    """For each fold: fit on the training part, then measure importance on the held-out part.
    Reports the mean drop in PR-AUC over folds and its spread across folds."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rng = np.random.default_rng(random_state)
    per_fold = {name: [] for name in groups}
    for train_idx, val_idx in skf.split(X, y):
        model = make_pipeline().fit(X.iloc[train_idx], y.iloc[train_idx])
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        for name, columns in groups.items():
            per_fold[name].append(permutation_drop(model, X_val, y_val, columns, n_repeats, rng))
    rows = [
        {
            "group": name,
            "n_columns": len(groups[name]),
            "mean_drop": float(np.mean(drops)),
            "std_across_folds": float(np.std(drops, ddof=1)),
            "min_fold": float(np.min(drops)),
            "max_fold": float(np.max(drops)),
        }
        for name, drops in per_fold.items()
    ]
    return pd.DataFrame(rows).sort_values("mean_drop", ascending=False).reset_index(drop=True)


def coefficient_table(fitted_pipeline) -> pd.DataFrame:
    """Coefficients of the fitted logistic pipeline, largest magnitude first.

    Numeric inputs are standardized (coefficient is per one standard deviation); categorical inputs are one-hot.
    Identical columns (for example every 'No internet service' column) share their weight equally,
    so a single coefficient understates the combined effect."""
    names = fitted_pipeline.named_steps["prep"].get_feature_names_out()
    coefs = fitted_pipeline.named_steps["model"].coef_[0]
    if len(names) != len(coefs):
        raise ValueError("feature names and coefficients have different lengths")
    order = np.argsort(-np.abs(coefs))
    table = pd.DataFrame({"feature": np.asarray(names)[order], "coefficient": coefs[order]})
    table["odds_ratio"] = np.exp(table["coefficient"])
    return table