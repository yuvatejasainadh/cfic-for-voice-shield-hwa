FROM python:3.11-slim

WORKDIR /app

# Install OpenMP runtime, system libraries, and compilation toolchain for llama.cpp CPU build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Pinned dependencies
COPY models/rias/requirements.txt ./requirements.txt

# Force CPU compilation for llama-cpp-python with OpenMP acceleration
ENV CMAKE_ARGS="-DGGML_OPENMP=ON" \
    FORCE_CMAKE=1

RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user and set up directory permissions
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && mkdir -p /app/weights \
    && chown -R appuser:appuser /app

COPY --chown=appuser:appuser models/rias/app ./app
COPY --chown=appuser:appuser models/rias/weights ./weights

USER appuser

EXPOSE 8331

ENV PYTHONUNBUFFERED=1 \
    GEMMA_MODEL_PATH=/app/weights/gemma-4-E2B-it-Q4_K_M.gguf \
    RIAS_MODEL_ID=google/gemma-4-E2B-it \
    RIAS_MODEL_REVISION=4adf15748de9e1a0097b6274cee9c15836c41c37 \
    RIAS_MODEL_SHA256=740185b21d22ceb83a11c3aa62ad5842ef32c70f6096d756bbee85a1e4ec34b8 \
    RIAS_USE_MOCK=false \
    RIAS_N_THREADS=4 \
    RIAS_N_CTX=2048 \
    RIAS_MAX_TOKENS=16 \
    RIAS_TIMEOUT_SECONDS=15.0 \
    RIAS_QUEUE_TIMEOUT_SECONDS=5.0 \
    RIAS_MAX_CONCURRENT_REQUESTS=2 \
    RIAS_MAX_QUEUE_SIZE=10

HEALTHCHECK --interval=20s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8331/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8331"]
