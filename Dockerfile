# Image that serves the fine-tuned DistilBERT model through FastAPI.
# Build it AFTER training (it copies models/distilbert-complaints/ into the image).
FROM python:3.11-slim

WORKDIR /app

# CPU-only PyTorch keeps the image much smaller than the default CUDA build
COPY requirements-api.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-api.txt

COPY app/ app/
COPY models/distilbert-complaints/ models/distilbert-complaints/

ENV MODEL_DIR=models/distilbert-complaints \
    HF_HUB_OFFLINE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
