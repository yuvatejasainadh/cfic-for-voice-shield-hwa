FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY models/pellav2/requirements.txt ./requirements.txt
RUN pip install --default-timeout=180 --retries 5 --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Create non-root user and set up model directories
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && mkdir -p /app/weights /app/config /home/appuser/.cache/huggingface \
    && chown -R appuser:appuser /app /home/appuser

COPY --chown=appuser:appuser models/pellav2/app ./app
COPY --chown=appuser:appuser models/pellav2/config ./config

USER appuser

EXPOSE 8321

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/home/appuser/.cache/huggingface \
    PELLAV2_MODEL_PATH=/app/weights/pellav2_k12_int8_final.pt \
    PELLAV2_CALIBRATION_PATH=/app/config/k12_calibration.json \
    PELLAV2_MODEL_VERSION=pellav2-k12-int8 \
    PELLAV2_USE_MOCK=false \
    PELLAV2_DEVICE=cpu \
    PELLAV2_THRESHOLD=0.50

HEALTHCHECK --interval=20s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8321/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8321"]
