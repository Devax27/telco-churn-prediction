"""Smoke tests for the Streamlit app (app.py), using Streamlit's AppTest and a small synthetic model."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.customer_form import ADDON_FIELDS
from tests.helpers import build_artifacts

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def keys(elements) -> set:
    return {element.key for element in elements}


@pytest.fixture
def at(tmp_path, monkeypatch):
    _df, _model, _card, model_path, card_path = build_artifacts(tmp_path)
    monkeypatch.setenv("CHURN_MODEL_PATH", str(model_path))
    monkeypatch.setenv("CHURN_CARD_PATH", str(card_path))
    return AppTest.from_file(APP, default_timeout=60).run()


def test_default_page_runs_and_shows_three_metrics(at):
    assert not at.exception
    assert [m.label for m in at.metric] == [
        "Churn probability",
        "Contact threshold",
        "Expected net benefit of contacting (illustrative units)",
    ]
    assert 0.0 <= float(at.metric[0].value.rstrip("%")) <= 100.0
    assert at.metric[1].value == "16.7%"   # 20 / (0.3 x 400)


def test_no_phone_service_hides_the_multiple_lines_choice(at):
    assert "multiple_lines" in keys(at.selectbox)
    at.selectbox(key="phone_service").set_value("No").run()
    assert not at.exception
    assert "multiple_lines" not in keys(at.selectbox)
    assert len(at.metric) == 3


def test_no_internet_service_hides_the_add_on_choices(at):
    assert set(ADDON_FIELDS) <= keys(at.selectbox)
    at.selectbox(key="internet_service").set_value("No").run()
    assert not at.exception
    assert not (set(ADDON_FIELDS) & keys(at.selectbox))
    assert len(at.metric) == 3


def test_manual_total_charges_shows_a_number_box(at):
    assert "total_charges" not in keys(at.number_input)
    at.checkbox(key="estimate_total").uncheck().run()
    assert not at.exception
    at.number_input(key="total_charges").set_value(1000.0).run()
    assert not at.exception
    assert len(at.metric) == 3


def test_a_very_expensive_offer_means_nobody_is_worth_contacting(at):
    at.sidebar.number_input(key="offer_cost").set_value(120.0).run()   # 120 / (0.3 x 400) = 1.0
    assert not at.exception
    assert at.metric[1].value == "100.0%"
    assert any("no customer" in info.value.lower() for info in at.info)


def test_missing_model_shows_a_helpful_error(tmp_path, monkeypatch):
    monkeypatch.setenv("CHURN_MODEL_PATH", str(tmp_path / "missing.joblib"))
    monkeypatch.setenv("CHURN_CARD_PATH", str(tmp_path / "missing.json"))
    page = AppTest.from_file(APP, default_timeout=60).run()
    assert not page.exception
    assert len(page.error) == 1
    assert "train_final" in page.error[0].value
    assert len(page.metric) == 0