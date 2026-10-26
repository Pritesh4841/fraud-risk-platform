"""
Fraud Risk Scoring API.  Run from project root:  uvicorn app:app --reload
Docs (Swagger UI): http://127.0.0.1:8000/docs
"""
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Literal, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

import src.preprocessing  # noqa: F401  (joblib needs it to unpickle build_features)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "fraud_model.joblib"
META_PATH = BASE_DIR / "models" / "metadata.json"

MERCHANT_CATEGORIES = ["Grocery", "Online Shopping", "Travel", "Electronics",
                       "Restaurant", "Fuel", "Jewellery"]
CATEGORY_LOOKUP = {c.lower(): c for c in MERCHANT_CATEGORIES}
CATEGORY_LOOKUP["online"] = "Online Shopping"          # alias used in the assignment's sample request
LOCATIONS = ["Ahmedabad", "Mumbai", "Delhi", "Bengaluru", "Pune", "Vadodara", "Surat",
             "Hyderabad", "Unknown/Foreign"]
LOCATION_LOOKUP = {c.lower(): c for c in LOCATIONS}

state = {"model": None, "meta": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load ONCE at startup (never retrain / reload per request)
    try:
        state["model"] = joblib.load(MODEL_PATH)
        state["meta"] = json.loads(META_PATH.read_text())
    except Exception as exc:  # keep API up so /health can report the problem
        print(f"[startup] could not load model: {exc}")
    yield


app = FastAPI(title="Fraud Risk Scoring API", version="1.0", lifespan=lifespan)


# ---------- request / response schemas (validation lives here) ----------
class Transaction(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # required
    amount: float = Field(..., strict=True, ge=0, le=10_000_000, allow_inf_nan=False)
    merchant_category: str = Field(..., strict=True, min_length=1)
    transaction_hour: int = Field(..., strict=True, ge=0, le=23)
    # optional (imputed exactly like in training if missing)
    customer_age: Optional[int] = Field(None, strict=True, ge=18, le=100)
    previous_transactions: Optional[int] = Field(None, strict=True, ge=0)
    failed_attempts: Optional[int] = Field(
        None, strict=True, ge=0,
        validation_alias=AliasChoices("failed_attempts", "previous_declines_24h"))
    location: Optional[str] = Field(None, strict=True)
    # Accepted + validated because the assignment's sample request sends it, but the dataset has no
    # equivalent column, so it is NOT used by the model (see README, "Field mapping").
    customer_age_days: Optional[int] = Field(None, strict=True, ge=0, le=100_000)

    @field_validator("merchant_category")
    @classmethod
    def check_category(cls, v: str) -> str:
        key = v.strip().lower()
        if key not in CATEGORY_LOOKUP:
            raise ValueError(f"must be one of {MERCHANT_CATEGORIES}")
        return CATEGORY_LOOKUP[key]

    @field_validator("location")
    @classmethod
    def clean_location(cls, v):
        if v is None or not v.strip():
            return None
        return LOCATION_LOOKUP.get(v.strip().lower(), v.strip())   # unknown city -> encoder ignores it


class Prediction(BaseModel):
    fraud_probability: float
    risk_band: Literal["LOW", "MEDIUM", "HIGH"]
    decision: Literal["APPROVE", "REVIEW"]


class BatchRequest(BaseModel):
    transactions: List[Transaction] = Field(..., min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    count: int
    predictions: List[Prediction]


# ---------- scoring ----------
def score(transactions: List[Transaction]) -> List[Prediction]:
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    def nan_if_none(x):
        return np.nan if x is None else x

    rows = [{
        "Amount": t.amount,
        "MerchantCategory": t.merchant_category,
        "Hour": t.transaction_hour,
        "CustomerAge": nan_if_none(t.customer_age),
        "PreviousTransactions": nan_if_none(t.previous_transactions),
        "FailedAttempts": nan_if_none(t.failed_attempts),
        "Location": nan_if_none(t.location),
    } for t in transactions]
    df = pd.DataFrame(rows)
    for col in ["CustomerAge", "PreviousTransactions", "FailedAttempts"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Location"] = df["Location"].astype(object)

    probs = state["model"].predict_proba(df)[:, 1]      # same pipeline as training
    high, medium = state["meta"]["decision_threshold"], state["meta"]["medium_threshold"]
    out = []
    for p in probs:
        band = "HIGH" if p >= high else "MEDIUM" if p >= medium else "LOW"
        out.append(Prediction(fraud_probability=round(float(p), 4), risk_band=band,
                              decision="REVIEW" if band == "HIGH" else "APPROVE"))
    return out


# ---------- endpoints ----------
@app.get("/health")
def health():
    loaded = state["model"] is not None
    meta = state["meta"] or {}
    return {
        "status": "ok" if loaded else "degraded",
        "model_loaded": loaded,
        "model_name": meta.get("model_name"),
        "thresholds": {"HIGH": meta.get("decision_threshold"), "MEDIUM": meta.get("medium_threshold")},
    }


@app.post("/predict", response_model=Prediction)
def predict(transaction: Transaction):
    return score([transaction])[0]


@app.post("/predict-batch", response_model=BatchResponse)
def predict_batch(batch: BatchRequest):
    preds = score(batch.transactions)
    return BatchResponse(count=len(preds), predictions=preds)