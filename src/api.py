"""API FastAPI para inferência; carrega o artefato joblib treinado."""
from typing import Literal
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config.paths_config import MODELS_DIR
from src.monitoring.live_metrics import record_prediction

ARTIFACT = MODELS_DIR / "stroke_pipeline.joblib"
app = FastAPI(title="Stroke Risk API", version="0.1.0", description="Estimativa experimental de risco de AVC.")
bundle = joblib.load(ARTIFACT) if ARTIFACT.exists() else None

class Patient(BaseModel):
    gender: Literal["Male", "Female", "Other"]
    age: float = Field(ge=0, le=120)
    hypertension: int = Field(ge=0, le=1)
    heart_disease: int = Field(ge=0, le=1)
    ever_married: Literal["No", "Yes"]
    work_type: Literal["children", "Govt_job", "Never_worked", "Private", "Self-employed"]
    Residence_type: Literal["Rural", "Urban"]
    avg_glucose_level: float = Field(gt=0)
    bmi: float | None = Field(default=None, ge=5, le=100)
    smoking_status: Literal["formerly smoked", "never smoked", "smokes", "Unknown"]

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": bundle is not None}

@app.get("/model")
def model_info():
    if bundle is None:
        raise HTTPException(status_code=503, detail="Modelo ausente. Execute python -m src.models.train.")
    return {"model": bundle["model_name"], "sampler": bundle["sampler"], "threshold": bundle["threshold"]}

@app.post("/predict")
def predict(patient: Patient):
    if bundle is None:
        raise HTTPException(status_code=503, detail="Modelo ausente. Execute python -m src.models.train.")
    features = pd.DataFrame([patient.model_dump()])
    probability = float(bundle["pipeline"].predict_proba(features)[0, 1])
    threshold = float(bundle["threshold"])
    prediction = int(probability >= threshold)
    record_prediction(prediction, bundle["model_name"])
    return {"stroke_probability": probability, "threshold": threshold,
            "prediction": prediction, "model": bundle["model_name"]}
