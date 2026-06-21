# AI-Agent Failure Predictor — container for the FastAPI scoring service.
# The model artefacts are gitignored and regenerable, so we train at build time
# (deterministic, seed 42, ~3.5 min) and bake champion.joblib + early_window.joblib
# into the image. The reproduction asserts in train.py fail the build on drift.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app

# Dependencies first (layer-cached). libgomp1 is needed by lightgbm/catboost wheels.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Source + config, then build the model into the image.
COPY src/ ./src/
COPY config/ ./config/
COPY models/model_card.md ./models/model_card.md
RUN python -m src.train --no-examples

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
