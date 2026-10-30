"""FastAPI service for the churn model.

Run from the project root:  uvicorn src.api:app --port 8000
Interactive docs:           http://127.0.0.1:8000/docs
"""
import logging
import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src import predict

logger = logging.getLogger("api")

YesNo = Literal["Yes", "No"]
AddOn = Literal["Yes", "No", "No internet service"]
ADDON_FIELDS = [
    "online_security", "online_backup", "device_protection",
    "tech_support", "streaming_tv", "streaming_movies",
]

# An invented customer for the interactive docs; not taken from the dataset.
EXAMPLE_CUSTOMER = {
    "gender": "Female", "senior_citizen": 0, "partner": "Yes", "dependents": "No", "tenure": 5,
    "phone_service": "Yes", "multiple_lines": "No", "internet_service": "Fiber optic",
    "online_security": "No", "online_backup": "No", "device_protection": "No", "tech_support": "No",
    "streaming_tv": "Yes", "streaming_movies": "No", "contract": "Month-to-month",
    "paperless_billing": "Yes", "payment_method": "Electronic check",
    "monthly_charges": 85.0, "total_charges": 425.0,
}


class CustomerFeatures(BaseModel):
    """One customer. Categories must match the training data exactly; unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE_CUSTOMER]})

    gender: Literal["Female", "Male"]
    senior_citizen: Literal[0, 1]
    partner: YesNo
    dependents: YesNo
    tenure: int = Field(ge=0, le=120, description="Months as a customer (training data covers 0 to 72)")
    phone_service: YesNo
    multiple_lines: Literal["Yes", "No", "No phone service"]
    internet_service: Literal["DSL", "Fiber optic", "No"]
    online_security: AddOn
    online_backup: AddOn
    device_protection: AddOn
    tech_support: AddOn
    streaming_tv: AddOn
    streaming_movies: AddOn
    contract: Literal["Month-to-month", "One year", "Two year"]
    paperless_billing: YesNo
    payment_method: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ]
    monthly_charges: float = Field(gt=0, le=500, description="Training data covers about 18 to 119")
    total_charges: float = Field(ge=0, le=100_000, description="Training data covers 0 to about 8,700")

    @model_validator(mode="after")
    def check_category_consistency(self):
        """The training data never contains contradictory combinations, so reject them."""
        if (self.phone_service == "No") != (self.multiple_lines == "No phone service"):
            raise ValueError("multiple_lines must be 'No phone service' exactly when phone_service is 'No'")
        no_internet = self.internet_service == "No"
        for name in ADDON_FIELDS:
            if no_internet != (getattr(self, name) == "No internet service"):
                raise ValueError(f"{name} must be 'No internet service' exactly when internet_service is 'No'")
        return self


class PredictionResponse(BaseModel):
    churn_probability: float = Field(description="Predicted probability that the customer leaves")
    threshold: float = Field(description="Contact customers whose probability is at least this value")
    recommend_contact: bool


def create_app(model_path=None, card_path=None) -> FastAPI:
    model_path = model_path or os.environ.get("CHURN_MODEL_PATH", predict.MODEL_PATH)
    card_path = card_path or os.environ.get("CHURN_CARD_PATH", predict.CARD_PATH)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = None
        app.state.card = None
        try:
            app.state.model, app.state.card = predict.load_artifacts(model_path, card_path)
            logger.info("model loaded: %s", app.state.card["model"])
        except predict.ModelLoadError as exc:
            logger.error("model not loaded: %s", exc)
        yield

    app = FastAPI(
        title="Telco churn prediction API",
        version="1.0.0",
        description=(
            "Method demonstration on a fictional dataset (IBM Telco churn sample). "
            "Returns a churn probability and a cost-based contact recommendation. "
            "The cost assumptions behind the threshold are illustrative."
        ),
        lifespan=lifespan,
    )

    @app.get("/health")
    def health(request: Request):
        return {"status": "ok", "model_loaded": request.app.state.model is not None}

    @app.get("/model-info")
    def model_info(request: Request):
        if request.app.state.card is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        return request.app.state.card

    @app.post("/predict", response_model=PredictionResponse)
    def predict_endpoint(customer: CustomerFeatures, request: Request):
        state = request.app.state
        if state.model is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        probability = predict.predict_churn_probability(state.model, customer.model_dump())
        threshold = state.card["threshold"]
        return PredictionResponse(
            churn_probability=probability,
            threshold=threshold,
            recommend_contact=predict.should_contact(probability, threshold),
        )

    return app


app = create_app()