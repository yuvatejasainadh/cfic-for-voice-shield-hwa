FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY services/transcription/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY services/transcription/service.py ./service.py

EXPOSE 8102

ENV TRANSCRIPTION_USE_MOCK=true
ENV TRANSCRIPTION_MODEL_VERSION=whisper-indic-mock-v0.1

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8102/health')" || exit 1

CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8102"]
