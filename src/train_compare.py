"""Compare candidate models by repeated cross-validation on the TRAINING set only.

The test set is not touched. Selection rule, written before looking at results: pick the highest mean
PR-AUC, unless its average gain over the reference model is below MIN_GAIN, in which case keep the
reference (simpler and easier to explain).

Run from the project root:  python src\\train_compare.py
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import modeling  # noqa: E402

OUT_PATH = ROOT / "docs" / "model_comparison_cv.csv"
SCORING = {"pr_auc": "average_precision", "roc_auc": "roc_auc", "neg_brier": "neg_brier_score"}
REFERENCE = "logistic regression (C=1.0)"
MIN_GAIN = 0.01

logger = logging.getLogger("train_compare")


def paired_summary(scores, reference) -> dict:
    """Compare two models fold by fold (both scored on identical folds)."""
    scores, reference = np.asarray(scores, dtype=float), np.asarray(reference, dtype=float)
    if scores.shape != reference.shape:
        raise ValueError("scores and reference must have the same number of folds")
    diff = scores - reference
    return {
        "mean_diff": float(diff.mean()),
        "std_diff": float(diff.std(ddof=1)) if diff.size > 1 else float("nan"),
        "win_rate": float((diff > 0).mean()),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    df = modeling.load_clean()
    X_train, X_test, y_train, y_test = modeling.split_data(df)
    logger.info("train: %s customers | test (not used here): %s customers", f"{len(X_train):,}", f"{len(X_test):,}")

    # The same splitter object gives identical folds for every model, so fold scores can be paired.
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=modeling.RANDOM_STATE)

    candidates = modeling.candidate_models()
    if REFERENCE not in candidates:
        raise ValueError(f"reference model {REFERENCE!r} is not among the candidates")

    all_scores = {}
    for name, pipeline in candidates.items():
        logger.info("evaluating: %s", name)
        # error_score="raise" makes a failing fold stop the script instead of silently becoming NaN.
        all_scores[name] = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=SCORING, error_score="raise")

    reference_pr = all_scores[REFERENCE]["test_pr_auc"]
    rows = []
    for name, s in all_scores.items():
        paired = paired_summary(s["test_pr_auc"], reference_pr)
        rows.append(
            {
                "model": name,
                "pr_auc_mean": s["test_pr_auc"].mean(),
                "pr_auc_std": s["test_pr_auc"].std(ddof=1),
                "roc_auc_mean": s["test_roc_auc"].mean(),
                "brier_mean": (-s["test_neg_brier"]).mean(),
                "fit_seconds_mean": s["fit_time"].mean(),
                "pr_auc_gain_vs_reference": paired["mean_diff"],
                "gain_std": paired["std_diff"],
                "fold_win_rate": paired["win_rate"],
            }
        )
    results = pd.DataFrame(rows).sort_values("pr_auc_mean", ascending=False).round(4)

    print()
    print(results.to_string(index=False))
    print(f"\nreference model: {REFERENCE}. gain = mean PR-AUC difference over the reference, fold by fold;")
    print("fold_win_rate = share of the 15 folds where the model beat the reference. Folds overlap, so")
    print("the spread is a rough guide, not a confidence interval.")

    best = results.iloc[0]
    print(f"\nhighest mean PR-AUC: {best['model']}")
    if best["model"] != REFERENCE and best["pr_auc_gain_vs_reference"] < MIN_GAIN:
        print(f"its gain over the reference is below {MIN_GAIN}, so the rule says: keep {REFERENCE}.")
    elif best["model"] != REFERENCE:
        print(f"its gain is at least {MIN_GAIN}, so the rule says: use {best['model']}.")
    else:
        print("the reference is already the best model.")

    results.to_csv(OUT_PATH, index=False)
    logger.info("wrote %s", OUT_PATH.name)


if __name__ == "__main__":
    main()