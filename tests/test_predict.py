"""Tests for src/predict.py: saving, loading, validation of the saved files, and prediction helpers."""
import json
import logging

import numpy as np
import pytest

from src import modeling, predict
from tests.helpers import build_artifacts


def test_save_and_load_round_trip_gives_identical_predictions(tmp_path):
    df, model, card, model_path, card_path = build_artifacts(tmp_path)
    loaded, loaded_card = predict.load_artifacts(model_path, card_path)
    X = df[modeling.FEATURES].head(50)
    assert np.allclose(model.predict_proba(X), loaded.predict_proba(X))
    assert loaded_card == card


def test_card_is_valid_json_with_the_expected_content(tmp_path):
    _df, _model, _card, _model_path, card_path = build_artifacts(tmp_path, threshold=0.25)
    card = json.loads(card_path.read_text(encoding="utf-8"))
    assert card["features"] == modeling.FEATURES
    assert card["threshold"] == pytest.approx(0.25)
    assert "scikit_learn" in card["versions"]
    assert "fictional" in card["intended_use"]


def test_missing_files_raise_a_clear_error(tmp_path):
    with pytest.raises(predict.ModelLoadError, match="not found"):
        predict.load_artifacts(tmp_path / "nope.joblib", tmp_path / "nope.json")


def test_corrupt_model_file_raises(tmp_path):
    _df, _model, _card, model_path, card_path = build_artifacts(tmp_path)
    model_path.write_bytes(b"this is not a model")
    with pytest.raises(predict.ModelLoadError, match="could not read"):
        predict.load_artifacts(model_path, card_path)


def test_feature_list_mismatch_raises(tmp_path):
    _df, _model, card, model_path, card_path = build_artifacts(tmp_path)
    card["features"] = card["features"][:-1]
    card_path.write_text(json.dumps(card), encoding="utf-8")
    with pytest.raises(predict.ModelLoadError, match="feature list"):
        predict.load_artifacts(model_path, card_path)


def test_version_mismatch_logs_a_warning(tmp_path, caplog):
    _df, _model, card, model_path, card_path = build_artifacts(tmp_path)
    card["versions"]["scikit_learn"] = "0.0.1"
    card_path.write_text(json.dumps(card), encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        predict.load_artifacts(model_path, card_path)
    assert "scikit-learn" in caplog.text


def test_predict_churn_probability_matches_predict_proba(tmp_path):
    df, model, _card, _mp, _cp = build_artifacts(tmp_path)
    record = df[modeling.FEATURES].iloc[0].to_dict()
    p = predict.predict_churn_probability(model, record)
    assert isinstance(p, float)
    assert 0.0 <= p <= 1.0
    assert p == pytest.approx(model.predict_proba(df[modeling.FEATURES].iloc[[0]])[0, 1])


def test_to_frame_rejects_missing_features():
    with pytest.raises(ValueError, match="missing features"):
        predict.to_frame([{"tenure": 5}])


def test_to_frame_ignores_extra_keys_and_orders_columns(tmp_path):
    df, *_ = build_artifacts(tmp_path)
    record = df[modeling.FEATURES].iloc[0].to_dict()
    shuffled = dict(reversed(list(record.items())))
    shuffled["ignored"] = "x"
    assert predict.to_frame([shuffled]).columns.tolist() == modeling.FEATURES


def test_should_contact_includes_the_threshold_itself():
    assert predict.should_contact(0.2, 0.2) is True
    assert predict.should_contact(0.1999, 0.2) is False
    assert predict.should_contact(0.9, 0.2) is True