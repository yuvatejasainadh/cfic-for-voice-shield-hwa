#!/usr/bin/env bash
# Run the full CFIC test suite across api-gateway, cloud-inference orchestrator, and model microservices
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "[1/6] Running API Gateway test suite..."
cd "${ROOT_DIR}/api-gateway"
pytest -v

echo ""
echo "[2/6] Running Cloud Inference Orchestrator test suite..."
cd "${ROOT_DIR}/cloud-inference/orchestrator"
pytest -v

echo ""
echo "[3/6] Running Open-Vakgyata LID microservice tests..."
cd "${ROOT_DIR}/cloud-inference/models/lid"
pytest -v tests

echo ""
echo "[4/6] Running English Whisper ASR microservice tests..."
cd "${ROOT_DIR}/cloud-inference/models/asr/indicconformer/english"
pytest -v tests

echo ""
echo "[5/6] Running Gemma RIAS LLM microservice tests..."
cd "${ROOT_DIR}/cloud-inference/models/rias"
pytest -v tests

echo ""
echo "[6/6] Running PellaV2 K12 INT8 acoustic detector microservice tests..."
cd "${ROOT_DIR}/cloud-inference/models/pellav2"
pytest -v tests

echo ""
echo "All test suites passed successfully!"
