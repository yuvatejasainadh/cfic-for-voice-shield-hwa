#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "[1/2] Running API Gateway test suite..."
cd "$ROOT_DIR/api-gateway"
python -m pytest -v

echo ""
echo "[2/2] Running Cloud Inference Orchestrator test suite..."
cd "$ROOT_DIR/cloud-inference/orchestrator"
python -m pytest -v

echo ""
echo "All test suites passed successfully!"
