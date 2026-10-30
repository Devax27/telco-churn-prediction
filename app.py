"""Streamlit demo for the churn model.

Run from the project root:  streamlit run app.py
Needs the saved model first:  python src\\train_final.py
"""
import os

import streamlit as st

from src import customer_form as cf
from src import decision, predict

MODEL_PATH = os.environ.get("CHURN_MODEL_PATH", str(predict.MODEL_PATH))
CARD_PATH = os.environ.get("CHURN_CARD_PATH", str(predict.CARD_PATH))

st.set_page_config(page_title="Telco churn demo", page_icon="📞", layout="wide")


@st.cache_resource
def load_model(model_path: str, card_path: str):
    return predict.load_artifacts(model_path, card_path)


def choose(label: str, options: list, key: str):
    """A selectbox that starts at the default customer's value."""
    return st.selectbox(label, options, index=options.index(cf.DEFAULTS[key]), key=key)


st.title("Telco churn: retention demo")
st.warning(
    "Method demonstration on a fictional dataset (IBM Telco churn sample). Not for decisions about real "
    "customers. The cost figures in the sidebar are illustrative assumptions."
)

try:
    model, card = load_model(MODEL_PATH, CARD_PATH)
except predict.ModelLoadError as exc:
    st.error(str(exc))
    st.stop()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Cost assumptions (illustrative)")
    offer_cost = st.number_input(
        "Cost of a retention offer", min_value=0.01, value=cf.DEFAULT_COSTS["offer_cost"], step=1.0, key="offer_cost"
    )
    value_saved = st.number_input(
        "Value of keeping a customer who would leave",
        min_value=0.01, value=cf.DEFAULT_COSTS["value_saved"], step=10.0, key="value_saved",
    )
    success_rate = st.slider(
        "Chance the offer keeps the customer", min_value=0.01, max_value=1.0,
        value=cf.DEFAULT_COSTS["success_rate"], step=0.01, key="success_rate",
    )
    threshold = decision.optimal_threshold(offer_cost, value_saved, success_rate)
    st.caption("Contact a customer when the predicted churn probability is at least offer cost / (chance x value).")

# --------------------------------------------------------------------- form
left, middle, right = st.columns(3)

with left:
    st.subheader("Customer")
    gender = choose("Gender", cf.OPTIONS["gender"], "gender")
    senior = st.checkbox("Senior citizen", value=bool(cf.DEFAULTS["senior_citizen"]), key="senior_citizen")
    partner = choose("Has a partner", cf.OPTIONS["partner"], "partner")
    dependents = choose("Has dependents", cf.OPTIONS["dependents"], "dependents")
    tenure = st.number_input(
        "Tenure (months)", min_value=0, max_value=72, value=int(cf.DEFAULTS["tenure"]), step=1, key="tenure",
        help="The training data covers 0 to 72 months.",
    )

with middle:
    st.subheader("Services")
    phone_service = choose("Phone service", cf.OPTIONS["phone_service"], "phone_service")
    if phone_service == "No":
        st.caption("Multiple lines: not applicable without phone service")
        multiple_lines = cf.NO_PHONE
    else:
        multiple_lines = choose("Multiple lines", cf.OPTIONS["multiple_lines"], "multiple_lines")
    internet_service = choose("Internet service", cf.OPTIONS["internet_service"], "internet_service")
    addons = {}
    if internet_service == "No":
        st.caption("Add-on services: not applicable without internet service")
        addons = {field: cf.NO_INTERNET for field in cf.ADDON_FIELDS}
    else:
        for field in cf.ADDON_FIELDS:
            addons[field] = choose(cf.ADDON_LABELS[field], cf.OPTIONS["addon"], field)

with right:
    st.subheader("Contract and billing")
    contract = choose("Contract", cf.OPTIONS["contract"], "contract")
    paperless = choose("Paperless billing", cf.OPTIONS["paperless_billing"], "paperless_billing")
    payment = choose("Payment method", cf.OPTIONS["payment_method"], "payment_method")
    monthly = st.number_input(
        "Monthly charges", min_value=18.0, max_value=120.0, value=float(cf.DEFAULTS["monthly_charges"]),
        step=0.05, key="monthly_charges", help="The training data covers about 18 to 119.",
    )
    estimate_total = st.checkbox(
        "Estimate total charges as tenure x monthly charges", value=True, key="estimate_total"
    )
    estimated = cf.estimate_total_charges(tenure, monthly)
    if estimate_total:
        total = estimated
        st.caption(f"Total charges: {estimated:,.2f}")
    else:
        total = st.number_input(
            "Total charges", min_value=0.0, max_value=10000.0, value=float(min(estimated, 10000.0)),
            step=1.0, key="total_charges",
        )

values = {
    "gender": gender, "senior_citizen": int(senior), "partner": partner, "dependents": dependents,
    "tenure": int(tenure), "phone_service": phone_service, "multiple_lines": multiple_lines,
    "internet_service": internet_service, **addons, "contract": contract, "paperless_billing": paperless,
    "payment_method": payment, "monthly_charges": float(monthly), "total_charges": float(total),
}
record = cf.build_record(values, estimate_total)
try:
    customer = cf.validate_record(record)
except ValueError as exc:
    st.error(f"These inputs are not a valid customer: {exc}")
    st.stop()

# ------------------------------------------------------------------ results
probability = predict.predict_churn_probability(model, customer.model_dump())
net_benefit = decision.expected_net_benefit(probability, offer_cost, value_saved, success_rate)

st.divider()
c1, c2, c3 = st.columns(3)
c1.metric("Churn probability", f"{100 * probability:.1f}%")
c2.metric("Contact threshold", f"{100 * threshold:.1f}%")
c3.metric("Expected net benefit of contacting (illustrative units)", f"{net_benefit:+,.1f}")

if threshold >= 1:
    st.info("At these costs no customer is worth contacting: an offer costs more than the most it can save.")
elif predict.should_contact(probability, threshold):
    st.success("Recommendation: contact this customer. The churn probability is at or above the threshold.")
else:
    st.info("Recommendation: do not contact this customer. The churn probability is below the threshold.")

st.caption(
    f"For comparison, {100 * card['train_churn_rate']:.1f}% of the customers the model was trained on "
    "left within the month."
)

with st.expander("About this model"):
    st.write(
        f"Model: {card['model']}, trained on {card['trained_on_rows']:,} customers "
        f"(saved {card['created_utc']}). Threshold stored with the model: {100 * card['threshold']:.1f}% "
        "(from the default cost assumptions). Predictions come from a logistic regression; they show "
        "associations in a fictional dataset, not causes."
    )