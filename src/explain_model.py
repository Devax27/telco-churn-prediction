"""Which inputs does the churn model rely on? Training data only; the test set is not touched.

Run from the project root:  python src\\explain_model.py
"""
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import explain, modeling  # noqa: E402

OUT_IMPORTANCE = ROOT / "docs" / "explanation_importance.csv"
OUT_COEFFICIENTS = ROOT / "docs" / "explanation_coefficients.csv"
IMG_PATH = ROOT / "docs" / "images" / "06_permutation_importance.png"

DUPLICATE_COLUMNS = [
    "cat__internet_service_No",
    "cat__online_security_No internet service",
    "cat__online_backup_No internet service",
    "cat__device_protection_No internet service",
    "cat__tech_support_No internet service",
    "cat__streaming_tv_No internet service",
    "cat__streaming_movies_No internet service",
    "cat__phone_service_No",
    "cat__multiple_lines_No phone service",
]

logger = logging.getLogger("explain_model")


def make_figure(importance) -> None:
    top = importance.head(15).iloc[::-1]
    colors = ["tab:orange" if g.startswith("[joint]") else "tab:blue" for g in top["group"]]
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    ax.barh(top["group"], top["mean_drop"], xerr=top["std_across_folds"], color=colors, capsize=3)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("drop in PR-AUC when the input is shuffled (mean over 5 folds; bars = spread across folds)")
    ax.set_title("What the churn model relies on (blue = single input, orange = related inputs shuffled together)")
    IMG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMG_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    df = modeling.load_clean()
    X_train, X_test, y_train, y_test = modeling.split_data(df)
    logger.info("train: %s customers | test (not used here): %s customers", f"{len(X_train):,}", f"{len(X_test):,}")

    importance = explain.cv_permutation_importance(
        modeling.build_logistic_pipeline, X_train, y_train, explain.feature_groups(), n_splits=5, n_repeats=10
    )
    print("\nCROSS-VALIDATED PERMUTATION IMPORTANCE (drop in PR-AUC; larger = the model relies on it more)")
    print(importance.round(4).to_string(index=False))
    print("\nA joint group shows how much the model loses when ALL its related inputs are shuffled together;")
    print("single inputs that are correlated with others understate their own importance.")

    model = modeling.build_logistic_pipeline().fit(X_train, y_train)
    coefficients = explain.coefficient_table(model)
    print("\nLARGEST COEFFICIENTS (numeric inputs standardized: per one standard deviation; others: one-hot)")
    print(coefficients.head(15).round(3).to_string(index=False))

    duplicates = coefficients[coefficients["feature"].isin(DUPLICATE_COLUMNS)]
    print("\nIDENTICAL COLUMNS SHARE THEIR WEIGHT (each is 1 for the same customers, so no single one shows the full effect)")
    print(duplicates.round(3).to_string(index=False))

    importance.round(5).to_csv(OUT_IMPORTANCE, index=False)
    coefficients.round(5).to_csv(OUT_COEFFICIENTS, index=False)
    make_figure(importance)
    logger.info("wrote %s, %s and %s", OUT_IMPORTANCE.name, OUT_COEFFICIENTS.name, IMG_PATH.name)


if __name__ == "__main__":
    main()