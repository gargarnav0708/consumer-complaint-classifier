"""REST API that predicts which financial product a consumer complaint is about.

Endpoints
    GET  /health   -> liveness / readiness check used by Kubernetes
    POST /predict  -> {"text": "..."}  =>  {"product": "...", "confidence": 0.97, "served_by": "<pod>"}

Run locally:  uvicorn app.main:app --port 8000
"""
import os
import socket
import threading

from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import pipeline

MODEL_DIR = os.getenv("MODEL_DIR", "models/distilbert-complaints")

# Load the model ONCE when the server starts, not on every request.
classifier = pipeline("text-classification", model=MODEL_DIR, tokenizer=MODEL_DIR)

# The tokenizer is not thread-safe, so each process runs one prediction at a time.
# Concurrency comes from running several replicas behind the Kubernetes Service.
model_lock = threading.Lock()

app = FastAPI(title="Consumer Complaint Classifier")


class Complaint(BaseModel):
    text: str = Field(min_length=1, examples=["I was charged a late fee on my credit card even though I paid on time."])


class Prediction(BaseModel):
    product: str
    confidence: float
    served_by: str  # pod name - shows the load being spread across replicas


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(complaint: Complaint):
    with model_lock:
        result = classifier(complaint.text, truncation=True, max_length=512)[0]
    return Prediction(
        product=result["label"],
        confidence=round(result["score"], 4),
        served_by=socket.gethostname(),
    )
