"""Baseline models, evaluated by cross-validation on the TRAINING set only.

The held-out test set is not touched here; it is used once, at the end of the project.

Run from the project root:  python src\\train_baseline.py
"""
import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import modeling  # noqa: E402

OUT_PATH = ROOT / "docs" / "baseline_cv_results.csv"
SCORING = {"pr_auc": "average_precision", "roc_auc": "roc_auc", "neg_brier": "neg_brier_score"}

logger = logging.getLogger("train_baseline")


def evaluate(name: str, pipeline, X, y, cv) -> dict:
    # error_score="raise" makes a failing fold stop the script instead of silently becoming NaN.
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=SCORING, error_score="raise")
    brier = -scores["test_neg_brier"]
    return {
        "model": name,
        "pr_auc_mean": scores["test_pr_auc"].mean(),
        "pr_auc_std": scores["test_pr_auc"].std(ddof=1),
        "roc_auc_mean": scores["test_roc_auc"].mean(),
        "roc_auc_std": scores["test_roc_auc"].std(ddof=1),
        "brier_mean": brier.mean(),
        "brier_std": brier.std(ddof=1),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    df = modeling.load_clean()
    X_train, X_test, y_train, y_test = modeling.split_data(df)
    logger.info("train: %s customers, churn rate %.2f%%", f"{len(X_train):,}", 100 * y_train.mean())
    logger.info("test (not used in this script): %s customers", f"{len(X_test):,}")

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=modeling.RANDOM_STATE)
    results = pd.DataFrame(
        [
            evaluate("baseline: predict the base rate", modeling.build_baseline_pipeline(), X_train, y_train, cv),
            evaluate("logistic regression (C=1.0)", modeling.build_logistic_pipeline(), X_train, y_train, cv),
        ]
    ).round(4)

    print()
    print(results.to_string(index=False))
    print("\nmean and standard deviation over 15 folds (5 folds x 3 repeats). Folds overlap, so the")
    print("spread is a rough guide, not a confidence interval. Higher PR-AUC and ROC-AUC are better;")
    print("lower Brier is better.")
    results.to_csv(OUT_PATH, index=False)
    logger.info("wrote %s", OUT_PATH.name)


if __name__ == "__main__":
    main()