FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for audio decoding if needed (e.g. ffmpeg, libsndfile1)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY services/pella/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY services/pella/service.py ./service.py

EXPOSE 8101

ENV PELLA_USE_MOCK=true
ENV PELLA_MODEL_VERSION=pellav2-scaffold-v0.1

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8101/health')" || exit 1

CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8101"]
