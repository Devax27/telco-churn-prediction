"""Shared test helpers: synthetic customers that follow the same category rules as the real data."""
import numpy as np
import pandas as pd

from src import clean_data, modeling, predict


def make_valid_df(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Random customers using only valid categories, with the consistency rules the cleaning step checks."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {col: rng.choice(sorted(values), size=n) for col, values in clean_data.ALLOWED_VALUES.items()}
    )
    df["multiple_lines"] = np.where(
        df["phone_service"] == "No", "No phone service", rng.choice(["Yes", "No"], size=n)
    )
    no_internet = (df["internet_service"] == "No").to_numpy()
    for col in clean_data.INTERNET_ADDON_COLUMNS:
        df[col] = np.where(no_internet, "No internet service", rng.choice(["Yes", "No"], size=n))

    tenure = rng.integers(0, 73, size=n)
    monthly = rng.uniform(20, 110, size=n).round(2)
    df["senior_citizen"] = rng.integers(0, 2, size=n)
    df["tenure"] = tenure
    df["monthly_charges"] = monthly
    df["total_charges"] = (tenure * monthly).round(2)
    prob = 1 / (1 + np.exp((tenure - 30) / 10))
    df[modeling.TARGET] = (rng.random(n) < prob).astype(int)
    return df


def build_artifacts(tmp_path, threshold: float = 0.2):
    """Train a small model on synthetic data and save it with a card under tmp_path."""
    df = make_valid_df()
    model = modeling.build_logistic_pipeline().fit(df[modeling.FEATURES], df[modeling.TARGET])
    card = predict.build_card("test model", len(df), df[modeling.TARGET].mean(), threshold, {"offer_cost": 1.0})
    model_path, card_path = tmp_path / "model.joblib", tmp_path / "card.json"
    predict.save_artifacts(model, card, model_path, card_path)
    return df, model, card, model_path, card_path