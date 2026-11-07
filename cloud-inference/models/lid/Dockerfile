FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install pinned dependencies
COPY models/lid/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Create non-root user and set up model cache directory
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && mkdir -p /app/weights /home/appuser/.cache/huggingface \
    && chown -R appuser:appuser /app /home/appuser

COPY --chown=appuser:appuser models/lid/app ./app

USER appuser

EXPOSE 8301

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/home/appuser/.cache/huggingface \
    LID_MODEL_ID=onecxi/open-vakgyata \
    LID_MODEL_REVISION=f2754058e485dfc65cc62589b8d0e21c1d328399 \
    LID_USE_MOCK=false \
    LID_DEVICE=auto

HEALTHCHECK --interval=20s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8301/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8301"]
