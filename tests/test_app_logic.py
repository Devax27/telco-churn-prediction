"""Tests for src/customer_form.py and the single-customer decision helper."""
from typing import get_args

import pytest

from src import api
from src import customer_form as cf
from src import decision
from src.threshold_analysis import BASE_CASE, SCENARIOS


def test_default_costs_match_the_base_case_scenario():
    assert cf.DEFAULT_COSTS == SCENARIOS[BASE_CASE]


def test_defaults_form_a_valid_customer():
    cf.validate_record(cf.build_record(cf.DEFAULTS))


def test_estimated_total_for_the_default_customer_equals_the_documented_example():
    assert cf.estimate_total_charges(5, 85.0) == pytest.approx(425.0)
    assert cf.DEFAULTS["total_charges"] == pytest.approx(425.0)


def test_no_phone_service_forces_the_multiple_lines_value():
    record = cf.build_record({**cf.DEFAULTS, "phone_service": "No", "multiple_lines": "Yes"})
    assert record["multiple_lines"] == "No phone service"
    cf.validate_record(record)


def test_no_internet_service_forces_the_add_on_values():
    record = cf.build_record({**cf.DEFAULTS, "internet_service": "No"})
    assert all(record[field] == "No internet service" for field in cf.ADDON_FIELDS)
    cf.validate_record(record)


def test_total_charges_are_estimated_only_when_asked():
    base = {**cf.DEFAULTS, "tenure": 10, "monthly_charges": 50.0, "total_charges": 123.0}
    assert cf.build_record(base, estimate_total=True)["total_charges"] == pytest.approx(500.0)
    assert cf.build_record(base, estimate_total=False)["total_charges"] == pytest.approx(123.0)


def test_build_record_does_not_modify_its_input():
    values = {**cf.DEFAULTS, "phone_service": "No", "multiple_lines": "Yes"}
    before = dict(values)
    cf.build_record(values)
    assert values == before


def test_a_contradictory_record_is_rejected():
    bad = {**cf.DEFAULTS, "phone_service": "No"}  # multiple_lines stays "No" instead of "No phone service"
    with pytest.raises(ValueError):
        cf.validate_record(bad)


def test_every_form_option_is_accepted_by_the_api_schema():
    fields = api.CustomerFeatures.model_fields
    for key, options in cf.OPTIONS.items():
        targets = cf.ADDON_FIELDS if key == "addon" else [key]
        for field in targets:
            assert set(options) <= set(get_args(fields[field].annotation)), field


def test_the_default_customer_values_are_among_the_form_options():
    for key, options in cf.OPTIONS.items():
        targets = cf.ADDON_FIELDS if key == "addon" else [key]
        for field in targets:
            assert cf.DEFAULTS[field] in options, field


# --------------------------------------------------- expected net benefit
def test_expected_net_benefit_hand_calculated():
    # offer 10, value 100, success 0.5: 0.5 * 0.5 * 100 - 10 = 15
    assert decision.expected_net_benefit(0.5, 10.0, 100.0, 0.5) == pytest.approx(15.0)


def test_expected_net_benefit_is_zero_at_the_theoretical_threshold():
    t = decision.optimal_threshold(10.0, 100.0, 0.5)
    assert decision.expected_net_benefit(t, 10.0, 100.0, 0.5) == pytest.approx(0.0, abs=1e-12)


def test_expected_net_benefit_matches_the_total_formula_for_one_customer():
    p = 0.37
    total = decision.net_benefit([p], [p], 0.0, offer_cost=10.0, value_saved=100.0, success_rate=0.5)
    assert decision.expected_net_benefit(p, 10.0, 100.0, 0.5) == pytest.approx(total)


def test_expected_net_benefit_rejects_an_invalid_probability():
    with pytest.raises(ValueError):
        decision.expected_net_benefit(1.2, 10.0, 100.0, 0.5)