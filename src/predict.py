"""Saving, loading and using the churn model, plus the model card that travels with it.

Security note: joblib files can execute code when loaded. Only load model files you created yourself.
"""
import json
import logging
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from src import modeling

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
CARD_PATH = ROOT / "models" / "model_card.json"

logger = logging.getLogger("predict")


class ModelLoadError(RuntimeError):
    """The saved model or its card is missing, unreadable, or inconsistent with this code."""


def build_card(model_name: str, n_train: int, train_churn_rate: float, threshold: float, costs: dict) -> dict:
    """Metadata saved next to the model."""
    return {
        "model": model_name,
        "trained_on_rows": int(n_train),
        "train_churn_rate": float(train_churn_rate),
        "features": list(modeling.FEATURES),
        "threshold": float(threshold),
        "cost_assumptions_illustrative": {k: float(v) for k, v in costs.items()},
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "intended_use": (
            "Method demonstration on a fictional dataset (IBM Telco churn sample). "
            "Not for decisions about real customers."
        ),
    }


def save_artifacts(model, card: dict, model_path=MODEL_PATH, card_path=CARD_PATH) -> None:
    model_path, card_path = Path(model_path), Path(card_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    card_path.write_text(json.dumps(card, indent=2), encoding="utf-8")


def load_artifacts(model_path=MODEL_PATH, card_path=CARD_PATH):
    """Load (model, card). Raises ModelLoadError if anything is missing, unreadable or inconsistent."""
    model_path, card_path = Path(model_path), Path(card_path)
    for path in (model_path, card_path):
        if not path.exists():
            raise ModelLoadError(f"{path} not found. Create it with: python src\\train_final.py")
    try:
        card = json.loads(card_path.read_text(encoding="utf-8"))
        model = joblib.load(model_path)
    except Exception as exc:  # unreadable or corrupt files
        raise ModelLoadError(f"could not read the model files: {exc}") from exc

    if card.get("features") != list(modeling.FEATURES):
        raise ModelLoadError("the model card's feature list differs from src/modeling.py; retrain the model")
    saved_version = card.get("versions", {}).get("scikit_learn")
    if saved_version != sklearn.__version__:
        logger.warning(
            "model was saved with scikit-learn %s but %s is installed; predictions may differ or fail",
            saved_version, sklearn.__version__,
        )
    return model, card


def to_frame(records) -> pd.DataFrame:
    """Turn a list of customer dicts into a table with exactly the model's columns, in order."""
    df = pd.DataFrame(list(records))
    missing = [c for c in modeling.FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"missing features: {missing}")
    return df[modeling.FEATURES]


def predict_churn_probability(model, record: dict) -> float:
    """Predicted probability that one customer (a dict of the 19 features) leaves."""
    return float(model.predict_proba(to_frame([record]))[0, 1])


def should_contact(probability: float, threshold: float) -> bool:
    """Same rule as the net-benefit analysis: contact when the probability is at least the threshold."""
    return probability >= threshold