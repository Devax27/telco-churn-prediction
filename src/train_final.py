"""Fit the chosen model on the TRAINING set, save it with a model card, and verify the saved file.

This is the model evaluated in Step 9 (logistic regression, C=1.0, trained on the 5,634 training customers),
so the reported test results describe exactly this file. The test set is not evaluated here.

Run from the project root:  python src\\train_final.py
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import decision, explain, modeling, predict  # noqa: E402
from src.threshold_analysis import BASE_CASE, SCENARIOS  # noqa: E402

COEF_PATH = ROOT / "docs" / "explanation_coefficients.csv"

logger = logging.getLogger("train_final")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    df = modeling.load_clean()
    X_train, _X_test, y_train, _y_test = modeling.split_data(df)
    logger.info("fitting on %s training customers (the test partition is not evaluated here)", f"{len(X_train):,}")
    model = modeling.build_logistic_pipeline(C=1.0).fit(X_train, y_train)

    costs = SCENARIOS[BASE_CASE]
    threshold = decision.optimal_threshold(**costs)
    card = predict.build_card("logistic regression (C=1.0)", len(X_train), y_train.mean(), threshold, costs)
    predict.save_artifacts(model, card, predict.MODEL_PATH, predict.CARD_PATH)
    logger.info(
        "saved %s (%.1f KB) and %s",
        predict.MODEL_PATH.name, predict.MODEL_PATH.stat().st_size / 1024, predict.CARD_PATH.name,
    )

    reloaded, _card = predict.load_artifacts()
    before = model.predict_proba(X_train)[:, 1]
    after = reloaded.predict_proba(X_train)[:, 1]
    if not np.allclose(before, after):
        raise ValueError("the reloaded model gives different predictions from the model that was saved")
    logger.info("OK: the reloaded model gives identical predictions on the training data")

    current = explain.coefficient_table(reloaded).set_index("feature")["coefficient"]
    reference = pd.read_csv(COEF_PATH).set_index("feature")["coefficient"]
    if set(current.index) != set(reference.index):
        raise ValueError("coefficient names differ from docs/explanation_coefficients.csv")
    max_diff = float((current.loc[reference.index] - reference).abs().max())
    if max_diff > 1e-4:
        raise ValueError(f"coefficients differ from Step 10's by up to {max_diff:.2e}")
    logger.info("OK: coefficients match docs/explanation_coefficients.csv (largest difference %.2e)", max_diff)
    logger.info("decision threshold stored in the card: %.4f", threshold)


if __name__ == "__main__":
    main()