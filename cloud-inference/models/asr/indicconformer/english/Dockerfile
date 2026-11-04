FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install pinned dependencies
COPY models/asr/indicconformer/english/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Create non-root user and set up model cache directory
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && mkdir -p /app/weights /home/appuser/.cache/huggingface \
    && chown -R appuser:appuser /app /home/appuser

COPY --chown=appuser:appuser models/asr/indicconformer/english/app ./app
COPY --chown=appuser:appuser models/asr/indicconformer/english/weights ./weights

USER appuser

EXPOSE 8311

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/home/appuser/.cache/huggingface \
    ASR_EN_MODEL_ID=openai/whisper-base.en \
    ASR_EN_MODEL_REVISION=911407f4214e0e1d82085af863093ec0b66f9cd6 \
    ASR_EN_USE_MOCK=false \
    ASR_EN_DEVICE=auto

HEALTHCHECK --interval=20s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8311/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8311"]
