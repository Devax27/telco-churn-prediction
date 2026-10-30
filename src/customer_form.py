"""Helpers shared by the Streamlit app and its tests: form options, defaults and record building."""
from src.api import ADDON_FIELDS, EXAMPLE_CUSTOMER, CustomerFeatures

NO_PHONE = "No phone service"
NO_INTERNET = "No internet service"

# Must equal the base-case scenario in src/threshold_analysis.py (a test checks this).
DEFAULT_COSTS = {"offer_cost": 20.0, "value_saved": 400.0, "success_rate": 0.3}

# The form starts from the same invented customer as the API's documentation example.
DEFAULTS = dict(EXAMPLE_CUSTOMER)

# Choices offered in the form. "addon" is shared by the six add-on services.
OPTIONS = {
    "gender": ["Female", "Male"],
    "partner": ["Yes", "No"],
    "dependents": ["No", "Yes"],
    "phone_service": ["Yes", "No"],
    "multiple_lines": ["No", "Yes"],
    "internet_service": ["DSL", "Fiber optic", "No"],
    "addon": ["No", "Yes"],
    "contract": ["Month-to-month", "One year", "Two year"],
    "paperless_billing": ["Yes", "No"],
    "payment_method": [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)",
    ],
}

ADDON_LABELS = {
    "online_security": "Online security",
    "online_backup": "Online backup",
    "device_protection": "Device protection",
    "tech_support": "Tech support",
    "streaming_tv": "Streaming TV",
    "streaming_movies": "Streaming movies",
}


def estimate_total_charges(tenure: float, monthly_charges: float) -> float:
    """Rough total billed to date: tenure x monthly charges (the real column is within about 2% of this)."""
    return round(float(tenure) * float(monthly_charges), 2)


def build_record(values: dict, estimate_total: bool = True) -> dict:
    """Apply the consistency rules the training data follows, and optionally estimate total charges.

    Returns a new dict; the input is not modified."""
    record = dict(values)
    if record["phone_service"] == "No":
        record["multiple_lines"] = NO_PHONE
    if record["internet_service"] == "No":
        for field in ADDON_FIELDS:
            record[field] = NO_INTERNET
    if estimate_total:
        record["total_charges"] = estimate_total_charges(record["tenure"], record["monthly_charges"])
    return record


def validate_record(record: dict) -> CustomerFeatures:
    """Validate with exactly the schema the API uses. Raises ValueError (pydantic's ValidationError) if invalid."""
    return CustomerFeatures(**record)