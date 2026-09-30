"""Maré Hotels' cancellation-risk service: the model behind `@champion`, with a value contract in front.

    uvicorn app:app --port 8001          # from this folder, in the notebook's environment

    GET  /health    is it up, and which model version is answering?
    POST /predict   one booking → P(cancel), whether to reconfirm it, and the model version

The model is whatever version the registry's `champion` alias points at when the service starts.
Moving the alias and restarting the service swaps the model; this file doesn't change.
"""

import os
from typing import Literal

import mlflow
import pandas as pd
from fastapi import FastAPI
from mlflow import MlflowClient
from pydantic import BaseModel, Field

REGISTERED_MODEL = "booking-cancellation"  # the model's name in the MLflow Model Registry
THRESHOLD = 0.5  # the revenue team reconfirms a booking at P(cancel) ≥ 0.5

NUMERIC = [
    "lead_time", "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children",
    "babies", "is_repeated_guest", "previous_cancellations", "previous_bookings_not_canceled",
    "days_in_waiting_list", "adr", "required_car_parking_spaces", "total_of_special_requests",
]
CATEGORICAL = [
    "hotel", "meal", "market_segment", "distribution_channel", "reserved_room_type",
    "deposit_type", "customer_type",
]
FEATURES = NUMERIC + CATEGORICAL


class Booking(BaseModel):
    """One booking, as the booking engine sends it: codes the model was trained on, counts ≥ 0."""

    booking_id: str
    hotel: Literal["City Hotel", "Resort Hotel"]
    lead_time: int = Field(ge=0)
    stays_in_weekend_nights: int = Field(ge=0)
    stays_in_week_nights: int = Field(ge=0)
    adults: int = Field(ge=0)
    children: int = Field(ge=0)
    babies: int = Field(ge=0)
    meal: Literal["BB", "HB", "FB", "SC", "Undefined"]
    market_segment: Literal[
        "Online TA", "Offline TA/TO", "Groups", "Direct", "Corporate", "Complementary", "Aviation", "Undefined"
    ]
    distribution_channel: Literal["TA/TO", "Direct", "Corporate", "GDS", "Undefined"]
    is_repeated_guest: Literal[0, 1]
    previous_cancellations: int = Field(ge=0)
    previous_bookings_not_canceled: int = Field(ge=0)
    reserved_room_type: Literal["A", "B", "C", "D", "E", "F", "G", "H", "L", "P"]
    deposit_type: Literal["No Deposit", "Non Refund", "Refundable"]
    customer_type: Literal["Transient", "Transient-Party", "Contract", "Group"]
    days_in_waiting_list: int = Field(ge=0)
    adr: float = Field(ge=0)
    required_car_parking_spaces: int = Field(ge=0)
    total_of_special_requests: int = Field(ge=0)


def load_champion():
    """Resolve `@champion` to a version number once, then load exactly that version."""
    version = MlflowClient().get_model_version_by_alias(REGISTERED_MODEL, "champion").version
    model = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL}/{version}")
    return model, version


mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
model, model_version = load_champion()  # once, at startup

app = FastAPI(title="Maré Hotels · cancellation risk")


@app.get("/health")
def health():
    """Up, and answering with this model version."""
    return {"status": "ok", "model": REGISTERED_MODEL, "model_version": model_version}


@app.post("/predict")
def predict(booking: Booking):
    """Score one booking. FastAPI has already validated the body against `Booking` (or answered 422)."""
    # one-row DataFrame in the model's column order; numbers as float64, which the signature expects
    frame = pd.DataFrame([booking.model_dump()])[FEATURES].astype({c: "float64" for c in NUMERIC})
    p_cancel = float(model.predict(frame)[0, 1])  # predict returns [P(kept), P(cancelled)] per row
    return {
        "booking_id": booking.booking_id,
        "p_cancel": round(p_cancel, 4),
        "reconfirm": p_cancel >= THRESHOLD,
        "threshold": THRESHOLD,
        "model_version": model_version,
    }
