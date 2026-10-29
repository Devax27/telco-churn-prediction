"""One-time evaluation of the chosen model on the held-out test set. See docs/final_evaluation_protocol.md.

The script refuses to overwrite its results file unless run with --rerun, so the test set is read once.
The model, features and threshold are fixed beforehand; nothing is tuned here.

Run from the project root:  python src\\final_evaluation.py
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import decision, modeling  # noqa: E402
from src.threshold_analysis import BASE_CASE, SCENARIOS  # noqa: E402

OUT_PATH = ROOT / "docs" / "final_test_results.csv"
CV_PATH = ROOT / "docs" / "model_comparison_cv.csv"
REFERENCE_MODEL = "logistic regression (C=1.0)"
N_BOOT = 2000
BOOT_SEED = 42

logger = logging.getLogger("final_evaluation")


def point_metrics(y, proba, threshold: float, costs: dict) -> dict:
    """Ranking metrics plus the outcome of contacting everyone with probability >= threshold."""
    y = np.asarray(y)
    p = np.asarray(proba)
    n = len(y)
    contacted = p >= threshold
    k = int(contacted.sum())
    caught = float(y[contacted].sum())
    churners = float(y.sum())
    nb = decision.net_benefit(y, p, threshold, **costs) / n * 1000
    nb_all = decision.net_benefit(y, p, 0.0, **costs) / n * 1000
    return {
        "pr_auc": average_precision_score(y, p),
        "roc_auc": roc_auc_score(y, p),
        "brier": brier_score_loss(y, p),
        "contacted_share": k / n,
        "precision": caught / k if k else np.nan,
        "recall": caught / churners,
        "net_benefit_per_1000": nb,
        "net_benefit_contact_all_per_1000": nb_all,
        "gain_vs_contact_all_per_1000": nb - nb_all,
    }


def bootstrap_metrics(y, proba, threshold: float, costs: dict, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> pd.DataFrame:
    """Metrics on resampled copies of the test set. Resamples with a single class are skipped (AUC is undefined)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    p = np.asarray(proba)
    n = len(y)
    rows = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        rows.append(point_metrics(yb, p[idx], threshold, costs))
    return pd.DataFrame(rows)


def summarise(estimates: dict, draws: pd.DataFrame) -> pd.DataFrame:
    low = draws.quantile(0.025)
    high = draws.quantile(0.975)
    return pd.DataFrame(
        [{"metric": m, "estimate": v, "ci_low": low[m], "ci_high": high[m]} for m, v in estimates.items()]
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="One-time test-set evaluation")
    parser.add_argument("--rerun", action="store_true", help="overwrite existing results (must be disclosed)")
    args = parser.parse_args()

    if OUT_PATH.exists() and not args.rerun:
        raise SystemExit(
            f"{OUT_PATH.name} already exists, so the test set has already been used. "
            "Use --rerun only if you will disclose the rerun in the README."
        )

    costs = SCENARIOS[BASE_CASE]
    threshold = decision.optimal_threshold(**costs)

    df = modeling.load_clean()
    X_train, X_test, y_train, y_test = modeling.split_data(df)
    logger.info("fitting the final model on %s training customers", f"{len(X_train):,}")
    model = modeling.build_logistic_pipeline(C=1.0).fit(X_train, y_train)

    logger.info("reading the held-out test set (%s customers) for the first evaluation", f"{len(X_test):,}")
    proba = model.predict_proba(X_test)[:, 1]
    y = y_test.to_numpy()

    estimates = point_metrics(y, proba, threshold, costs)
    draws = bootstrap_metrics(y, proba, threshold, costs)
    logger.info("bootstrap: used %d of %d resamples", len(draws), N_BOOT)
    table = summarise(estimates, draws)

    print(f"\nTEST SET: {len(y):,} customers, {int(y.sum())} churners, churn rate {100 * y.mean():.2f}%")
    print(f"threshold {threshold:.4f} (base case: offer cost {costs['offer_cost']}, value saved {costs['value_saved']}, "
          f"success rate {costs['success_rate']}; illustrative assumptions)\n")
    print(table.round(4).to_string(index=False))

    calib = decision.calibration_table(y, proba, n_bins=5)
    shown = calib.copy()
    for col in ["mean_predicted", "observed_rate", "ci_low", "ci_high"]:
        shown[col] = (100 * shown[col]).round(1)
    print("\nCALIBRATION ON THE TEST SET (5 groups of predicted risk, percentages)")
    print(shown.to_string())

    cv = pd.read_csv(CV_PATH)
    cv_pr = float(cv.loc[cv["model"] == REFERENCE_MODEL, "pr_auc_mean"].iloc[0])
    pr = table.set_index("metric").loc["pr_auc"]
    gain = table.set_index("metric").loc["gain_vs_contact_all_per_1000"]
    print(f"\nFLAG 1: cross-validation PR-AUC was {cv_pr:.4f}; test interval {pr['ci_low']:.4f} to {pr['ci_high']:.4f}")
    if cv_pr < pr["ci_low"] or cv_pr > pr["ci_high"]:
        print("  -> the interval excludes the CV estimate: investigate for split or leakage errors (do not retune).")
    else:
        print("  -> consistent with the cross-validation estimate.")
    print(f"FLAG 2: net benefit minus contact-everyone, per 1,000 customers: {gain['estimate']:,.0f} "
          f"(95% interval {gain['ci_low']:,.0f} to {gain['ci_high']:,.0f})")
    if gain["ci_low"] <= 0:
        print("  -> the advantage over contacting everyone is not statistically clear on this test set.")
    else:
        print("  -> the advantage over contacting everyone is clear (interval above zero).")

    table.round(6).to_csv(OUT_PATH, index=False)
    logger.info("wrote %s", OUT_PATH.name)


if __name__ == "__main__":
    main()