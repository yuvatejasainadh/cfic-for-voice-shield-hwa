#!/usr/bin/env bash
# Shell runner for Whisper benchmark
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

MODEL="${1:-small}"
MANIFEST="${2:-data/manifest_telugu.jsonl}"
OUTPUT="${3:-outputs/results/whisper_telugu.csv}"
CHUNK_OUTPUT="${4:-outputs/results/chunk_results.csv}"

echo "Running Whisper ($MODEL) benchmark..."
.venv-whisper/bin/python src/run_whisper.py \
    --manifest "$MANIFEST" \
    --model "$MODEL" \
    --output "$OUTPUT" \
    --chunk-output "$CHUNK_OUTPUT"
