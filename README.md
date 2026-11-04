# CloudForge Intelligence Core (CFIC) × VoiceShield HackVersion

**Developed by Team SAIL for HackWithAditya**  
**Repository Lead & System Architect:** Kasukurthi Yuvateja Sainadh  
**Architecture Model:** Decoupled HYBRID Architecture (API Gateway + Cloud Inference)

---

## 1. Overview

**CloudForge Intelligence Core (CFIC)** is the backend intelligence system powering **VoiceShield**, a real-time defense platform protecting mobile and enterprise voice calls against AI-generated deepfake voices and conversational fraud.

CFIC decouples application-level API lifecycle management from model-serving workloads into two isolated subsystems:
1. **API Gateway (`api-gateway/`)**: Public edge gateway handling authentication, rate limiting, audio payload validation, 5-second chunking orchestration, application-level risk aggregation, decision management, and normalized client responses.
2. **Cloud Inference (`cloud-inference/`)**: Dedicated model-serving subsystem handling Language Identification (LID), language-specific IndicConformer ASR checkpoints, and dual acoustic spoof detection models (VMC-South fine-tuned AASIST-L and AASIST-L fallback).

---

## 2. HYBRID Architecture

```
                      VoiceShield Client Audio Payload
                                     │
                                     ▼ (HTTP POST /v1/analyze, X-API-Key)
                ┌─────────────────────────────────────────┐
                │          CFIC API Gateway (8000)        │
                │  - X-API-Key Authentication             │
                │  - Sliding Window Rate Limiter (30/60s) │
                │  - Max Audio Validation (≤ 25 MiB)      │
                │  - X-Request-ID Distributed Tracing     │
                │  - 5-Second Window Chunk Orchestration  │
                └────────────────────┬────────────────────┘
                                     │ (Private Network)
                                     ▼
                ┌─────────────────────────────────────────┐
                │     Cloud Inference Orchestrator (8200) │
                │  - Step 1: Language Identification      │
                │  - Step 2: Dynamic Checkpoint Routing   │
                │  - Step 3: Concurrent Model Execution   │
                └──────────────┬───────────────────┬──────┘
                               │                   │
                 ┌─────────────┴─────┐       ┌─────┴──────────────┐
                 ▼                   ▼       ▼                    ▼
          ┌─────────────┐     ┌─────────────┐ ┌────────────┐ ┌─────────────┐
          │  Model LID  │     │ Indic-      │ │ VMC-South  │ │  AASIST-L   │
          │   (8301)    │     │ Conformer   │ │ (AASIST-L  │ │  Fallback   │
          │   Identify  │     │ (EN/TE/HI)  │ │ South-Ind) │ │  (English)  │
          │   Language  │     │ (8311-8313) │ │   (8321)   │ │   (8322)    │
          └─────────────┘     └─────────────┘ └────────────┘ └─────────────┘
                                     │                   │
                                     └─────────┬─────────┘
                                               │ (Unified Inference Response)
                                               ▼
                ┌─────────────────────────────────────────┐
                │      API Gateway Decision Engine        │
                │  - Conversational Risk Synthesis (RIAS) │
                │  - Mandatory HIGH Risk Override         │
                │  - Fail-Safe Inconclusive Policy        │
                └────────────────────┬────────────────────┘
                                     │
                                     ▼
                ┌─────────────────────────────────────────┐
                │            Client Response              │
                │   (GENUINE / FAKE / SUSPICIOUS /        │
                │            INCONCLUSIVE)                │
                └─────────────────────────────────────────┘
```

---

## 3. Subsystem Responsibilities

### 3.1 API Gateway Responsibilities
- Public REST API endpoints (`POST /v1/analyze`, `GET /health`).
- API Key authentication (`X-API-Key`) and authorization.
- Sliding-window in-memory rate limiting (30 requests per 60 seconds).
- Audio payload validation (MIME types, size limit ≤ 25 MiB).
- Temporary audio storage and 5-second chunking orchestration.
- Application-level job lifecycle management.
- Application-level semantic risk aggregation.
- Decision Engine evaluation and normalized client responses.

### 3.2 Cloud Inference Responsibilities
- Spoken Language Identification (LID).
- Dynamic ASR checkpoint selection:
  - English IndicConformer (`indicconformer-en`)
  - Telugu IndicConformer (`indicconformer-te`)
  - Hindi IndicConformer (`indicconformer-hi`)
- Dynamic acoustic spoof model selection:
  - **VMC-South**: Fine-tuned AASIST-L for South Indian regional speech (Telugu, Tamil, Kannada, Malayalam).
  - **AASIST-L Fallback**: Standard AASIST-L for English and non-regional audio.
- Audio normalization and 16 kHz mono resampling.
- Latency measurement and model result normalization.

---

## 4. Models in the Architecture

| Model Role | Microservice | Model Checkpoint | Target Domain / Language | License | Container Port |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Language ID (LID)** | model-lid | onecxi/open-vakgyata (Wav2Vec2) | 10 Indian & English languages | CC-BY-NC-4.0 | 8301 |
| **English ASR** | model-asr-en | openai/whisper-base.en | Dedicated English Speech-to-Text | Apache-2.0 | 8311 |
| **Telugu ASR** | model-asr-te | AI4Bharat IndicConformer | Telugu (	e) | CC-BY-4.0 | 8312 |
| **Hindi ASR** | model-asr-hi | AI4Bharat IndicConformer | Hindi (hi) | CC-BY-4.0 | 8313 |
| **South Indian Spoof** | model-vmc-south | AASIST-L (Fine-tuned) | South Indian Languages (	e, 	a, kn, ml) | Academic | 8321 |
| **Fallback Spoof** | model-aasist-fallback | AASIST-L Baseline | English & General Fallback | Apache-2.0 | 8322 |

> Detailed technical specifications, benchmark results, and licensing information are available in [LID & English ASR Model Documentation](docs/architecture/models-lid-and-english-asr.md).

---

## 5. Repository Layout

```text
CFIC-VoiceShield-HackVersion/
│
├── api-gateway/                      # API Gateway Subsystem
│   ├── app/
│   │   ├── api/routes/               # Route endpoints (/health, /analyze)
│   │   ├── core/                     # Config, security, errors, rate limiter
│   │   ├── middleware/               # Request tracing (X-Request-ID)
│   │   ├── db/                       # Database session & models scaffold
│   │   ├── schemas/                  # Pydantic request & response schemas
│   │   ├── services/                 # Auth, audio, cfic_client, job, risk
│   │   ├── orchestration/            # Analysis orchestrator & decision engine
│   │   └── main.py                   # Gateway FastAPI entry point
│   ├── alembic/                      # Database migrations scaffold
│   ├── tests/                        # Gateway test suite
│   ├── requirements.txt              # Decoupled gateway dependencies
│   ├── Dockerfile                    # Gateway container build
│   └── .env.example                  # Environment configuration template
│
├── cloud-inference/                  # Cloud Inference Subsystem
│   ├── orchestrator/                 # Inference coordinator
│   │   ├── app/                      # API, routing, services, main.py
│   │   ├── tests/                    # Orchestrator test suite
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   ├── models/                       # Isolated model microservices
│   │   ├── lid/                      # Language Identification service
│   │   ├── asr/indicconformer/       # Language-specific ASR
│   │   │   ├── english/
│   │   │   ├── hindi/
│   │   │   └── telugu/
│   │   ├── vmc-south/                # Fine-tuned AASIST-L for South Indian speech
│   │   └── aasist-fallback/          # AASIST-L fallback / English detector
│   └── shared/                       # Contracts, audio preprocessing, schemas
│       ├── contracts/                # Inter-service markdown contracts
│       ├── audio/                    # 16 kHz mono preprocessing utilities
│       └── schemas/                  # Inter-service Pydantic schemas
│
├── infrastructure/                   # Infrastructure configuration
│   ├── redis/                        # Distributed rate limiting config
│   ├── postgres/                     # DB initialization script
│   ├── monitoring/                   # Prometheus monitoring config
│   └── docker/                       # Docker environment files
│
├── legacy/                           # Archived prototype components
│   ├── services/                     # Old PellaV2, Transcription, RIAS
│   └── tced/                         # Prototype TCED policy
│
├── docs/                             # Architecture & operational docs
│   ├── architecture/                 # Hybrid architecture & legacy mapping
│   ├── api-contracts/                # Gateway & cloud inference contracts
│   ├── security/                     # Auth & rate-limiting specifications
│   └── deployment/                   # Docker compose & setup guides
│
├── cfic-asr-benchmark/               # Standalone ASR benchmark (IndicSynth)
├── scripts/                          # Operational helper scripts
├── docker-compose.yml                # Unified multi-container orchestration
├── .gitignore
└── README.md
```

---

## 6. Local Development

### 6.1 Setup API Gateway
```powershell
# Activate Python 3.11/3.12 virtual environment
.\.venv\Scripts\Activate.ps1

# Navigate to API Gateway and configure environment
cd api-gateway
copy .env.example .env

# Run API Gateway locally
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- Swagger API Documentation: `http://localhost:8000/docs`
- Health Endpoint: `http://localhost:8000/health`

### 6.2 Setup Cloud Inference Orchestrator
```powershell
cd cloud-inference/orchestrator
uvicorn app.main:app --reload --host 0.0.0.0 --port 8200
```

---

## 7. Docker Deployment

Deploy the entire HYBRID microservice cluster with Docker Compose:

```bash
docker compose up --build -d
```

- **API Gateway:** Available publicly on port `8000`.
- **Cloud Inference Subsystem:** Isolated on the private bridge network `cfic-internal-network`.
- **Model Services:** Load weights mounted via read-only Docker volumes (`/app/weights:ro`).

---

## 8. Environment Configuration

Copy `api-gateway/.env.example` to `api-gateway/.env` (and root `.env` for Docker Compose):

| Variable | Description | Default |
| :--- | :--- | :--- |
| `CFIC_ENV` | Application environment (`development`, `production`) | `development` |
| `CFIC_API_KEY` | Secret API key required in `X-API-Key` header | `replace-me-for-local-testing` |
| `CFIC_RATE_LIMIT_REQUESTS` | Maximum allowed requests per window | `30` |
| `CFIC_RATE_LIMIT_WINDOW_SECONDS` | Rate limit window in seconds | `60` |
| `CFIC_MAX_AUDIO_BYTES` | Maximum allowed audio upload size (bytes) | `26214400` (25 MiB) |
| `CFIC_CLOUD_INFERENCE_URL` | Internal Cloud Inference Orchestrator URL | `http://cloud-inference-orchestrator:8200` |
| `LID_USE_MOCK` | Enable mock scaffold for LID model | `true` |
| `ASR_EN_USE_MOCK` | Enable mock scaffold for English ASR | `true` |
| `VMC_SOUTH_USE_MOCK` | Enable mock scaffold for VMC-South detector | `true` |

---

## 9. Testing

```powershell
# Run API Gateway test suite (47 baseline tests)
cd api-gateway
python -m pytest -v

# Run Cloud Inference Orchestrator tests
cd ..\cloud-inference\orchestrator
python -m pytest -v
```

---

## 10. Team Contribution Workflow

| Contributor | Focus Area | Subsystem Directory | Recommended Feature Branch |
| :--- | :--- | :--- | :--- |
| **Sainadh** | System Architect | `api-gateway/`, `docker-compose.yml` | `main` |
| **Varun** | MLOps Architect | `cloud-inference/models/vmc-south/`, `aasist-fallback/` | `feature/varun-spoof-models` |
| **Varshu** | Cloud Engineer | `cloud-inference/orchestrator/`, `cloud-inference/models/lid/` | `feature/varshu-inference-orchestrator` |
| **Pallavi** | Cloud Engineer | `cloud-inference/models/asr/indicconformer/` | `feature/pallavi-indicconformer-asr` |

For component migration history, see [Legacy Component Mapping](docs/architecture/legacy-component-mapping.md).
