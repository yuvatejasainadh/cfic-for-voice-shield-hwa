#!/usr/bin/env bash
# Shell runner for IndicConformer benchmark
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

MODEL_PATH="${1:-models/indicconformer_stt_te_hybrid_rnnt_large.nemo}"
DECODER="${2:-ctc}"
MANIFEST="${3:-data/manifest_telugu.jsonl}"
OUTPUT="${4:-outputs/results/indicconformer_telugu.csv}"
CHUNK_OUTPUT="${5:-outputs/results/chunk_results.csv}"

echo "Running IndicConformer benchmark (decoder: $DECODER)..."
.venv-indic/bin/python src/run_indicconformer.py \
    --manifest "$MANIFEST" \
    --nemo-model "$MODEL_PATH" \
    --decoder "$DECODER" \
    --output "$OUTPUT" \
    --chunk-output "$CHUNK_OUTPUT"
