# Local Development Setup Guide

## 1. Prerequisites

- Python 3.11 or 3.12
- Git
- Docker & Docker Compose (optional for local non-containerized testing)

---

## 2. API Gateway Local Setup

```powershell
# 1. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 2. Install gateway dependencies
cd api-gateway
pip install -r requirements.txt
pip install pytest pytest-asyncio httpx

# 3. Configure environment
copy .env.example .env

# 4. Run API Gateway
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## 3. Running Tests

```powershell
# Run API Gateway tests
cd api-gateway
python -m pytest -v

# Run Cloud Inference Orchestrator tests
cd ..\cloud-inference\orchestrator
python -m pytest -v
```
