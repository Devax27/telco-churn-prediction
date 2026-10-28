"""Cost-based decision threshold for the churn model, using TRAINING data only.

Pre-registered rule (written before running): for the base-case scenario, use the theoretical threshold
t* = offer_cost / (success_rate * value_saved). Also report the threshold that maximises the empirical net
benefit. If the net benefit at t* is more than MAX_GAP_PCT below that maximum, flag a calibration problem.

All cost figures are illustrative assumptions for a fictional dataset, not real business costs.
The test set is not used.

Run from the project root:  python src\\threshold_analysis.py
"""
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import decision, modeling  # noqa: E402

OUT_CSV = ROOT / "docs" / "threshold_analysis.csv"
IMG_PATH = ROOT / "docs" / "images" / "05_threshold_analysis.png"

BASE_CASE = "B base case"
MAX_GAP_PCT = 5.0
GRID = np.round(np.arange(0.0, 1.0001, 0.01), 2)

SCENARIOS = {
    "A cheap offer": dict(offer_cost=10.0, value_saved=400.0, success_rate=0.3),
    "B base case": dict(offer_cost=20.0, value_saved=400.0, success_rate=0.3),
    "C expensive offer": dict(offer_cost=50.0, value_saved=400.0, success_rate=0.3),
    "D low success rate": dict(offer_cost=20.0, value_saved=400.0, success_rate=0.1),
}

logger = logging.getLogger("threshold_analysis")


def summarise(name: str, costs: dict, y, proba):
    """Return the full threshold table and a one-row summary for one cost scenario."""
    t_star = decision.optimal_threshold(**costs)
    table = decision.threshold_table(y, proba, GRID, **costs)
    best = table.loc[table["net_benefit_per_1000"].idxmax()]
    at_star = decision.threshold_table(y, proba, [t_star], **costs).iloc[0]
    contact_all = float(table.loc[table["threshold"] == 0.0, "net_benefit_per_1000"].iloc[0])

    best_value = float(best["net_benefit_per_1000"])
    star_value = float(at_star["net_benefit_per_1000"])
    gap = 100 * (best_value - star_value) / best_value if best_value > 0 else np.nan
    row = {
        "scenario": name,
        **costs,
        "theory_threshold": t_star,
        "best_grid_threshold": float(best["threshold"]),
        "best_net_benefit_per_1000": best_value,
        "net_benefit_at_theory_per_1000": star_value,
        "gap_pct": gap,
        "contacted_share_at_theory": float(at_star["contacted_share"]),
        "precision_at_theory": float(at_star["precision"]),
        "recall_at_theory": float(at_star["recall"]),
        "contact_everyone_per_1000": contact_all,
    }
    return table, row


def make_figure(calib: pd.DataFrame, tables: dict, summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    ax = axes[0]
    err = np.vstack(
        [
            (calib["observed_rate"] - calib["ci_low"]).clip(lower=0),
            (calib["ci_high"] - calib["observed_rate"]).clip(lower=0),
        ]
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfect calibration")
    ax.errorbar(calib["mean_predicted"], calib["observed_rate"], yerr=err, fmt="o", capsize=3,
                label="deciles of predicted risk (95% interval)")
    ax.set_xlabel("mean predicted churn probability")
    ax.set_ylabel("observed churn rate")
    ax.set_title("Calibration (out-of-fold, training customers)")
    ax.legend(loc="upper left")

    ax = axes[1]
    for _, row in summary.iterrows():
        table = tables[row["scenario"]]
        line, = ax.plot(table["threshold"], table["net_benefit_per_1000"], label=row["scenario"])
        ax.axvline(row["theory_threshold"], color=line.get_color(), linestyle=":", alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("contact customers with predicted churn probability at least this value")
    ax.set_ylabel("net benefit per 1,000 customers (illustrative units)")
    ax.set_title("Net benefit by threshold (dotted line = theoretical threshold)")
    ax.legend()

    IMG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMG_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    df = modeling.load_clean()
    X_train, X_test, y_train, y_test = modeling.split_data(df)
    logger.info("train: %s customers | test (not used here): %s customers", f"{len(X_train):,}", f"{len(X_test):,}")

    # The model chosen in Step 7 by the pre-registered rule: logistic regression, C=1.0.
    proba = decision.out_of_fold_proba(modeling.build_logistic_pipeline(C=1.0), X_train, y_train)
    logger.info(
        "out-of-fold predictions (averaged over 3 repeats): PR-AUC %.4f | ROC-AUC %.4f | Brier %.4f",
        average_precision_score(y_train, proba), roc_auc_score(y_train, proba), brier_score_loss(y_train, proba),
    )
    logger.info("training churn rate: %.2f%%", 100 * y_train.mean())

    calib = decision.calibration_table(y_train, proba, n_bins=10)
    shown = calib.copy()
    for col in ["mean_predicted", "observed_rate", "ci_low", "ci_high"]:
        shown[col] = (100 * shown[col]).round(1)
    print("\nCALIBRATION (deciles of predicted risk, percentages)")
    print(shown.to_string())
    gap = (calib["observed_rate"] - calib["mean_predicted"]).abs()
    print(f"\nlargest gap between predicted and observed rate in any decile: {100 * gap.max():.1f} points")

    tables, rows = {}, []
    for name, costs in SCENARIOS.items():
        tables[name], row = summarise(name, costs, y_train.to_numpy(), proba)
        rows.append(row)
    summary = pd.DataFrame(rows)

    print("\nSCENARIOS (cost figures are illustrative assumptions; net benefit is per 1,000 customers)")
    print(summary.round(3).to_string(index=False))

    base = summary.loc[summary["scenario"] == BASE_CASE].iloc[0]
    print(f"\nPRE-REGISTERED RULE, base case: use threshold {base['theory_threshold']:.3f}")
    print(f"net benefit there: {base['net_benefit_at_theory_per_1000']:,.0f} per 1,000 customers; "
          f"empirical best (threshold {base['best_grid_threshold']:.2f}): {base['best_net_benefit_per_1000']:,.0f}")
    if base["gap_pct"] > MAX_GAP_PCT:
        print(f"gap {base['gap_pct']:.1f}% is above {MAX_GAP_PCT}%: calibration problem flagged, investigate before using it.")
    else:
        print(f"gap {base['gap_pct']:.1f}% is within {MAX_GAP_PCT}%: the theoretical threshold is acceptable.")

    summary.round(4).to_csv(OUT_CSV, index=False)
    make_figure(calib, tables, summary)
    logger.info("wrote %s and %s", OUT_CSV.name, IMG_PATH.name)


if __name__ == "__main__":
    main()