# Docker Compose Deployment Guide

## 1. Quick Start

Run the entire HYBRID stack with Docker Compose:

```bash
# 1. Copy environment template
copy api-gateway\.env.example .env    # Windows
cp api-gateway/.env.example .env      # Linux/macOS

# 2. Build and launch services
docker compose up --build -d
```

---

## 2. Exposed Ports & Access

| Service | Host Port | Accessibility |
| :--- | :--- | :--- |
| **API Gateway** | `8000` | **Public** (client-facing entry point) |
| **Cloud Inference Orchestrator** | *Internal* | **Private** (`cfic-internal-network`) |
| **Model Services** | *Internal* | **Private** (`cfic-internal-network`) |

---

## 3. Verifying Health

```bash
# Verify API Gateway
curl http://localhost:8000/health
```

---

## 4. Volume Mounts for Model Checkpoints

Model weights are mapped into containers using read-only volume mounts:
- `./cloud-inference/models/lid/weights:/app/weights:ro`
- `./cloud-inference/models/asr/indicconformer/english/weights:/app/weights:ro`
- `./cloud-inference/models/asr/indicconformer/telugu/weights:/app/weights:ro`
- `./cloud-inference/models/asr/indicconformer/hindi/weights:/app/weights:ro`
- `./cloud-inference/models/vmc-south/weights:/app/weights:ro`
- `./cloud-inference/models/aasist-fallback/weights:/app/weights:ro`
