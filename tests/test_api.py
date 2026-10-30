"""Tests for src/api.py using FastAPI's TestClient and a small model trained on synthetic data."""
from typing import get_args

import pytest
from fastapi.testclient import TestClient

from src import api, clean_data, modeling, predict
from tests.helpers import build_artifacts


@pytest.fixture
def served(tmp_path):
    _df, model, card, model_path, card_path = build_artifacts(tmp_path, threshold=0.2)
    app = api.create_app(model_path=model_path, card_path=card_path)
    with TestClient(app) as client:
        yield client, model, card


def payload(**overrides) -> dict:
    return {**api.EXAMPLE_CUSTOMER, **overrides}


NO_INTERNET = {"internet_service": "No", **{f: "No internet service" for f in api.ADDON_FIELDS}}


def test_health_reports_a_loaded_model(served):
    client, _model, _card = served
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_model_info_returns_the_card(served):
    client, _model, card = served
    body = client.get("/model-info").json()
    assert body["threshold"] == pytest.approx(0.2)
    assert body["features"] == card["features"]


def test_predict_returns_probability_threshold_and_decision(served):
    client, _model, _card = served
    response = client.post("/predict", json=payload())
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"churn_probability", "threshold", "recommend_contact"}
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["threshold"] == pytest.approx(0.2)
    assert body["recommend_contact"] == (body["churn_probability"] >= body["threshold"])


def test_api_result_equals_the_direct_function_result(served):
    client, model, _card = served
    expected = predict.predict_churn_probability(model, payload())
    assert client.post("/predict", json=payload()).json()["churn_probability"] == pytest.approx(expected)


def test_missing_field_is_rejected(served):
    client, _m, _c = served
    body = payload()
    del body["contract"]
    assert client.post("/predict", json=body).status_code == 422


def test_unknown_field_is_rejected(served):
    client, _m, _c = served
    assert client.post("/predict", json=payload(extra_field=1)).status_code == 422


def test_invalid_category_is_rejected(served):
    client, _m, _c = served
    assert client.post("/predict", json=payload(contract="Three year")).status_code == 422


@pytest.mark.parametrize(
    "field, value",
    [
        ("tenure", -1), ("tenure", 121), ("monthly_charges", 0), ("monthly_charges", 501),
        ("senior_citizen", 2), ("total_charges", -1),
    ],
)
def test_out_of_range_values_are_rejected(served, field, value):
    client, _m, _c = served
    assert client.post("/predict", json=payload(**{field: value})).status_code == 422


def test_wrong_type_is_rejected(served):
    client, _m, _c = served
    assert client.post("/predict", json=payload(tenure="abc")).status_code == 422


def test_inconsistent_phone_fields_are_rejected_but_consistent_ones_accepted(served):
    client, _m, _c = served
    for bad in [
        dict(phone_service="No", multiple_lines="Yes"),
        dict(phone_service="Yes", multiple_lines="No phone service"),
    ]:
        response = client.post("/predict", json=payload(**bad))
        assert response.status_code == 422
        assert "multiple_lines" in response.text
    ok = client.post("/predict", json=payload(phone_service="No", multiple_lines="No phone service"))
    assert ok.status_code == 200


def test_inconsistent_internet_fields_are_rejected_but_consistent_ones_accepted(served):
    client, _m, _c = served
    assert client.post("/predict", json=payload(internet_service="No")).status_code == 422
    assert client.post("/predict", json=payload(tech_support="No internet service")).status_code == 422
    assert client.post("/predict", json=payload(**NO_INTERNET)).status_code == 200


def test_request_schema_has_exactly_the_model_features():
    assert set(api.CustomerFeatures.model_fields) == set(modeling.FEATURES)


def test_request_categories_match_the_cleaning_rules():
    for column, allowed in clean_data.ALLOWED_VALUES.items():
        annotation = api.CustomerFeatures.model_fields[column].annotation
        assert set(get_args(annotation)) == set(allowed), column


def test_documentation_example_is_a_valid_customer():
    api.CustomerFeatures(**api.EXAMPLE_CUSTOMER)


def test_service_without_a_model_reports_it_and_returns_503(tmp_path):
    app = api.create_app(model_path=tmp_path / "missing.joblib", card_path=tmp_path / "missing.json")
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok", "model_loaded": False}
        assert client.post("/predict", json=payload()).status_code == 503
        assert client.get("/model-info").status_code == 503