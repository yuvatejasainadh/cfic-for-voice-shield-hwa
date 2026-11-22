#!/usr/bin/env bash
# Linux/macOS setup script for CFIC x VoiceShield ASR Benchmark
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

echo "Setting up isolated virtual environments for CFIC x VoiceShield ASR Benchmark..."

# 1. Whisper environment
echo "[1/2] Setting up Whisper environment (.venv-whisper)..."
if [ ! -d ".venv-whisper" ]; then
    python3 -m venv .venv-whisper
fi
.venv-whisper/bin/pip install --upgrade pip
.venv-whisper/bin/pip install -r requirements-whisper.txt

# 2. IndicConformer environment
echo "[2/2] Setting up IndicConformer environment (.venv-indic)..."
if [ ! -d ".venv-indic" ]; then
    python3 -m venv .venv-indic
fi
.venv-indic/bin/pip install --upgrade pip
.venv-indic/bin/pip install -r requirements-indicconformer.txt

# Apply NeMo compatibility patches
.venv-indic/bin/python src/patch_nemo.py

echo "Setup completed successfully."
