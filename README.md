# CloudForge Intelligence Core (CFIC) × VoiceShield HackVersion

**Developed by The SAILors for HackWithAditya 2026**

**Team:** The SAILors  
**Organization:** SAI Labs  
**Repository Lead & System Architect:** Kasukurthi Yuvateja Sainadh  
**Architecture:** Decoupled Hybrid Microservices Architecture  
**Core Components:** API Gateway + Cloud Inference + Model Microservices  
**Deployment:** Docker Compose

---

# 1. Project Overview

## 1.1 CloudForge Intelligence Core

CloudForge Intelligence Core (CFIC) is the backend intelligence system powering **VoiceShield**, an AI-powered voice security platform designed to detect AI-generated deepfake voices and conversational fraud.

CFIC provides a modular, decoupled architecture that separates application-level request management from computationally intensive machine learning inference.

The system is divided into two primary subsystems:

1. **API Gateway (`api-gateway/`)**
   - Public-facing API.
   - Authentication and authorization.
   - Rate limiting.
   - Audio validation and preprocessing.
   - Five-second audio chunking.
   - Job lifecycle management and queuing.
   - Final risk aggregation and decision management.

2. **Cloud Inference (`cloud-inference/`)**
   - Language Identification (LID).
   - Language-specific Automatic Speech Recognition (ASR).
   - Acoustic spoof/deepfake detection using PellaV2.
   - Conversational risk assessment using Gemma RIAS.
   - Model routing, inference coordination and response normalization.

The architecture allows individual models to be developed, containerized, tested and optimized independently.

## 1.2 Core Objectives

- Detect AI-generated or manipulated speech.
- Identify the spoken language automatically.
- Transcribe supported Indian languages accurately.
- Detect conversational fraud, impersonation and social-engineering attempts.
- Combine acoustic and semantic intelligence for a unified risk assessment.
- Support scalable, containerized deployment.
- Maintain privacy, security and modularity.

---

# 2. Updated System Architecture

## 2.1 Architectural Design

CFIC follows a decoupled hybrid architecture consisting of:

- API Gateway for external client communication.
- Cloud Inference Orchestrator for model coordination.
- Independent model-serving microservices.
- Centralized application-level decision engine.

The architecture uses two independent detection paths:

**Path A — Acoustic Intelligence**

Analyzes the audio signal to identify potential AI-generated or manipulated speech using PellaV2.

**Path B — Conversational Intelligence**

Identifies the spoken language, generates a transcript and evaluates conversational fraud risk using Gemma RIAS.

Both paths contribute to the final VoiceShield decision.

## 2.2 Updated Architecture Diagram

```text
                 VOICESHIELD CLIENT
                  Audio Input
                       |
                       v
        +--------------------------------+
        |       CFIC API GATEWAY         |
        |             :8000              |
        |--------------------------------|
        | API Authentication             |
        | Rate Limiting                  |
        | Audio Validation               |
        | Request ID / Job Management    |
        | 5-Second Audio Chunking        |
        | Queue Management               |
        +---------------+----------------+
                        |
                        v
        +--------------------------------+
        |  CLOUD INFERENCE ORCHESTRATOR  |
        |             :8200              |
        +---------------+----------------+
                        |
             +----------+----------+
             |                     |
             v                     v
    +----------------+    +--------------------+
    | ACOUSTIC PATH  |    | CONVERSATIONAL PATH|
    +----------------+    +--------------------+
             |                     |
             v                     v
    +----------------+    +--------------------+
    |    PellaV2     |    | Language ID (LID)  |
    | Acoustic Deep- |    |      :8301         |
    | fake Detection |    +---------+----------+
    |     :8321      |              |
    +--------+-------+              v
             |              Language Routing
             |                     |
             |          +----------+----------+
             |          |          |          |
             |          v          v          v
             |       Telugu     Hindi     English
             |       Indic      Indic     Whisper
             |       Conformer  Conformer  ASR
             |       :8312      :8313      :8311
             |          |          |          |
             |          +----------+----------+
             |                     |
             |                     v
             |              Transcript
             |                     |
             |                     v
             |             +---------------+
             |             |   Gemma RIAS  |
             |             |     :8331     |
             |             | Conversational|
             |             | Risk Analysis |
             |             +-------+-------+
             |                     |
             +----------+----------+
                        |
                        v
        +--------------------------------+
        |       CFIC DECISION ENGINE     |
        |--------------------------------|
        | Acoustic Risk Evaluation       |
        | Conversational Risk Evaluation |
        | Risk Aggregation               |
        | Mandatory High-Risk Override   |
        | Fail-Safe Inconclusive Policy  |
        +---------------+----------------+
                        |
                        v
        +--------------------------------+
        |       NORMALIZED RESPONSE      |
        |                                |
        | GENUINE / FAKE / SUSPICIOUS /  |
        | INCONCLUSIVE                   |
        +---------------+----------------+
                        |
                        v
                  VOICESHIELD CLIENT
```

*Note: The diagram represents the target integration architecture. Individual services must be independently validated before being marked operational.*

## 2.3 Architectural Principles

1. Each model operates as an independently deployable service.
2. The orchestrator owns inference routing and coordination.
3. The API Gateway owns application-level request lifecycle and final response management.
4. LID determines the appropriate ASR checkpoint.
5. Acoustic spoof detection and conversational fraud detection remain independent inference tasks.
6. Gemma RIAS analyzes transcripts, not raw audio.
7. The final decision engine combines the available inference results.
8. Internal services communicate through private Docker networking.
9. Model weights are mounted independently from application source code.
10. Model failures must not silently produce false genuine classifications.

---

# 3. System Components and Responsibilities

## 3.1 API Gateway

**Directory:** `api-gateway/`

The API Gateway is the public entry point of CFIC.

Responsibilities:

- Expose REST API endpoints.
- Authenticate requests using API keys.
- Enforce request rate limits.
- Validate audio formats and payload sizes.
- Generate and propagate request IDs.
- Store temporary audio data securely.
- Divide long audio inputs into five-second windows.
- Manage inference jobs and queues.
- Communicate with the Cloud Inference Orchestrator.
- Collect acoustic and conversational inference results.
- Execute the final decision engine.
- Return normalized client responses.

### Security and request management

| Feature | Specification |
|---|---|
| Authentication | X-API-Key |
| Rate limiting | 30 requests / 60 seconds |
| Audio upload limit | 25 MiB |
| Chunking | 5-second windows |
| Request tracing | X-Request-ID |
| Queue management | Job-based processing |
| Error handling | Structured API responses |

The API Gateway must not directly load or execute machine learning checkpoints.

## 3.2 Cloud Inference Orchestrator

**Directory:** `cloud-inference/orchestrator/`

The orchestrator coordinates independent model services.

Responsibilities:

- Accept inference jobs from the API Gateway.
- Forward audio to PellaV2.
- Invoke LID.
- Determine language-specific ASR routing.
- Coordinate ASR execution.
- Aggregate transcript segments.
- Forward normalized transcripts to Gemma RIAS.
- Collect model outputs.
- Normalize inference responses.
- Handle model timeouts and service failures.

The orchestrator must not duplicate the API Gateway's authentication, rate-limiting or application-level job management responsibilities.

## 3.3 Language Identification (LID)

**Directory:** `cloud-inference/models/lid/`

**Model:** Open-Vakgyata

Responsibilities:

- Identify the spoken language from audio.
- Support the configured Indian language set.
- Return language codes and confidence scores.
- Provide the language decision to the orchestrator.
- Enable language-specific ASR checkpoint selection.

The LID service must remain independent of ASR services.

## 3.4 Language-Specific ASR

**Directory:** `cloud-inference/models/asr/indicconformer/`

CFIC uses language-specific transcription checkpoints instead of depending exclusively on a single multilingual transcription model.

| Language | Model | Service | Port |
|---|---|---|---|
| English | Whisper Base English | model-asr-en | 8311 |
| Telugu | IndicConformer Telugu | model-asr-te | 8312 |
| Hindi | IndicConformer Hindi | model-asr-hi | 8313 |

Responsibilities:

- Accept normalized audio input.
- Perform speech-to-text inference.
- Return transcription and model metadata.
- Support chunk-based transcription.
- Maintain consistent response contracts.

### Routing policy

| LID Result | ASR Selection |
|---|---|
| Telugu (`te`) | Telugu IndicConformer |
| Hindi (`hi`) | Hindi IndicConformer |
| English (`en`) | Whisper Base English |
| Unsupported language | Structured unsupported-language response |
| Low confidence | Inconclusive or configured retry policy |

Additional language-specific checkpoints may be integrated in future releases without redesigning the overall architecture.

## 3.5 Acoustic Spoof Detection — PellaV2

**Directory:** `cloud-inference/models/pellav2/`

**Responsible:** Varun

PellaV2 is the acoustic intelligence component of VoiceShield.

Responsibilities:

- Analyze audio signals for potential synthetic or manipulated speech.
- Produce structured acoustic detection results.
- Support the selected quantized model format.
- Expose inference latency and model metadata.
- Operate as an independent microservice.

### Model optimization

Varun is responsible for:

- Evaluating the original PellaV2 checkpoint.
- Comparing available quantized variants.
- Measuring model size and inference performance.
- Validating detection consistency after quantization.
- Packaging the selected checkpoint for containerized inference.
- Documenting the final model format and checksum.

The selected quantized model must be evaluated against the original checkpoint before being locked.

**Important architecture decision:** VMC-South and its associated AASIST-L fallback architecture are removed from the current project plan. PellaV2 is the designated acoustic detection component.

## 3.6 Conversational Risk Assessment — Gemma RIAS

**Directory:** `cloud-inference/models/rias/`

**Selected Model:** `google/gemma-4-E2B-it`

**Quantized Format:** GGUF Q4_K_M

**Inference Runtime:** llama.cpp / llama-cpp-python

**Service Port:** 8331

Gemma RIAS evaluates the semantic content of transcribed conversations.

Responsibilities:

- Analyze conversational context.
- Identify impersonation attempts.
- Detect social-engineering patterns.
- Identify urgency and intimidation tactics.
- Detect financial manipulation and fraudulent instructions.
- Generate structured risk assessments.

### Locked model details

| Property | Selected Configuration |
|---|---|
| Model | Gemma 4 E2B IT |
| Quantization | GGUF Q4_K_M |
| Checkpoint size | Approximately 2.96 GB |
| Docker image | Approximately 1.01 GB |
| Runtime | llama.cpp |
| Execution | CPU-based, OpenMP |
| Container | model-rias-gemma |
| Port | 8331 |

The Gemma RIAS service has already been containerized and validated.

**Qwen is not part of the selected architecture.**

---

# 4. Final Decision Engine

The CFIC Decision Engine combines independent model outputs into one application-level risk decision.

## 4.1 Input Signals

1. PellaV2 acoustic detection result.
2. LID language identification result.
3. ASR transcript.
4. Gemma RIAS conversational risk assessment.
5. Service execution status and confidence metadata.

## 4.2 Decision Categories

| Final Classification | Meaning |
|---|---|
| GENUINE | No significant acoustic or conversational threat detected |
| FAKE | Strong evidence of synthetic or manipulated speech |
| SUSPICIOUS | Significant conversational fraud or other suspicious behavior |
| INCONCLUSIVE | Insufficient evidence or critical inference failure |

## 4.3 Decision Principles

- A confirmed high-risk signal must not be downgraded by a low-risk result from another model.
- Acoustic spoof detection and conversational fraud detection must remain distinguishable in the response.
- Missing model outputs must not be interpreted as safe results.
- Critical inference failures must trigger the configured fail-safe policy.
- The system must preserve individual model results for auditing and debugging.
- The final decision must be deterministic and documented.

The exact confidence thresholds and aggregation rules must be established through validation rather than invented.

---

# 5. Repository Structure

```text
CFIC-VoiceShield-HackVersion/
│
├── api-gateway/
│   ├── app/
│   │   ├── api/routes/
│   │   ├── core/
│   │   ├── middleware/
│   │   ├── db/
│   │   ├── schemas/
│   │   ├── services/
│   │   ├── orchestration/
│   │   └── main.py
│   ├── tests/
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
│
├── cloud-inference/
│   │
│   ├── orchestrator/
│   │   ├── app/
│   │   ├── tests/
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   │
│   ├── models/
│   │   │
│   │   ├── lid/
│   │   │   ├── app/
│   │   │   ├── tests/
│   │   │   ├── weights/
│   │   │   └── Dockerfile
│   │   │
│   │   ├── asr/
│   │   │   └── indicconformer/
│   │   │       ├── english/
│   │   │       ├── hindi/
│   │   │       └── telugu/
│   │   │
│   │   ├── pellav2/
│   │   │   ├── app/
│   │   │   ├── tests/
│   │   │   ├── weights/
│   │   │   └── Dockerfile
│   │   │
│   │   └── rias/
│   │       ├── app/
│   │       ├── tests/
│   │       ├── weights/
│   │       └── Dockerfile
│   │
│   └── shared/
│       ├── contracts/
│       ├── audio/
│       └── schemas/
│
├── infrastructure/
│   ├── redis/
│   ├── postgres/
│   ├── monitoring/
│   └── docker/
│
├── legacy/
│   ├── services/
│   └── tced/
│
├── docs/
│   ├── architecture/
│   ├── api-contracts/
│   ├── security/
│   ├── deployment/
│   └── models/
│
├── cfic-asr-benchmark/
│
├── scripts/
│
├── docker-compose.yml
├── .gitignore
└── README.md
```

The actual directory layout should be reconciled with the existing repository during implementation. This is the intended logical organization, not authorization to create duplicate directories unnecessarily.

---

# 6. Docker and Deployment Architecture

CFIC uses Docker Compose to coordinate independently containerized services.

## 6.1 Service Ports

| Component | Internal Port |
|---|---:|
| API Gateway | 8000 |
| Cloud Inference Orchestrator | 8200 |
| LID | 8301 |
| English Whisper ASR | 8311 |
| Telugu IndicConformer | 8312 |
| Hindi IndicConformer | 8313 |
| PellaV2 | 8321 |
| Gemma RIAS | 8331 |

These are the intended service ports and must be reconciled with the actual implementation.

## 6.2 Networking

- API Gateway exposes the public application API.
- Cloud Inference services operate on a private Docker network.
- Internal service communication uses Docker DNS names.
- Model services should not expose unnecessary host ports.
- Model weights are mounted read-only wherever possible.
- Docker build contexts must exclude large model binaries.

## 6.3 Model Packaging

Model checkpoints must remain separate from application source code.

Preferred storage:

- Hugging Face repositories for model distribution where appropriate.
- GitHub repository for application code, configuration and deployment files.
- Versioned model references and SHA-256 checksums for reproducibility.

Large model binaries must not be committed directly into the main application repository unless explicitly justified.

---

# 7. Environment Configuration

The following represents the intended configuration categories. Exact variable names must match the actual service implementations.

| Variable | Purpose |
|---|---|
| `CFIC_ENV` | Application environment |
| `CFIC_API_KEY` | API authentication |
| `CFIC_RATE_LIMIT_REQUESTS` | Maximum requests |
| `CFIC_RATE_LIMIT_WINDOW_SECONDS` | Rate limit window |
| `CFIC_MAX_AUDIO_BYTES` | Maximum audio payload |
| `CFIC_CLOUD_INFERENCE_URL` | Orchestrator endpoint |
| `LID_USE_MOCK` | LID development mock |
| `ASR_EN_USE_MOCK` | English ASR development mock |
| `ASR_TE_USE_MOCK` | Telugu ASR development mock |
| `ASR_HI_USE_MOCK` | Hindi ASR development mock |
| `PELLAV2_USE_MOCK` | PellaV2 development mock |
| `RIAS_USE_MOCK` | RIAS development mock |

Production or final validation configurations must disable mocks.

All secrets must be supplied through environment configuration and must never be committed.

---

# 8. Team Organization and Task Distribution

## Team: The SAILors

**Organization:** SAI Labs

The team follows a modular ownership model in which each contributor owns a defined engineering responsibility.

## 8.1 Sainadh — Repository Lead & System Architect

**Primary responsibility:** Overall architecture, API Gateway and full-system integration.

Assigned work:

1. Own the overall CFIC architecture.
2. Maintain the Software Requirements Specification.
3. Maintain API Gateway architecture and implementation.
4. Implement API authentication and authorization.
5. Implement rate limiting.
6. Implement audio validation and five-second chunking.
7. Implement job lifecycle and queue management.
8. Integrate LID with the inference orchestrator.
9. Coordinate language-specific ASR routing.
10. Integrate PellaV2 and Gemma RIAS.
11. Implement application-level decision aggregation.
12. Maintain Docker Compose and service networking.
13. Perform integration testing and regression testing.
14. Review pull requests and approve architectural changes.
15. Maintain repository quality and release readiness.

**Ownership directories:**

- `api-gateway/`
- `cloud-inference/orchestrator/` (integration and architecture)
- `cloud-inference/models/lid/` (LID integration and coordination)
- `docker-compose.yml`
- `docs/architecture/`

**Branch:** `feat/sainadh-api-integration`

## 8.2 Pallavi — Telugu ASR Engineer

**Primary responsibility:** Telugu IndicConformer containerization and validation.

Assigned work:

1. Inspect the selected Telugu IndicConformer checkpoint.
2. Prepare the model-serving application.
3. Implement the inference API.
4. Build the Dockerfile.
5. Configure model weight mounting.
6. Optimize CPU/GPU inference configuration as supported.
7. Implement input and output validation.
8. Test Telugu transcription accuracy.
9. Validate five-second chunk inference.
10. Document model requirements and deployment instructions.

**Ownership directory:**

`cloud-inference/models/asr/indicconformer/telugu/`

**Branch:** `feat/pallavi-telugu-asr`

## 8.3 Varshu — Hindi ASR Engineer

**Primary responsibility:** Hindi IndicConformer containerization and validation.

Assigned work:

1. Inspect the selected Hindi IndicConformer checkpoint.
2. Prepare the model-serving application.
3. Implement the inference API.
4. Build the Dockerfile.
5. Configure model weight mounting.
6. Optimize inference configuration.
7. Implement input and output validation.
8. Test Hindi transcription accuracy.
9. Validate five-second chunk inference.
10. Document model requirements and deployment instructions.

**Ownership directory:**

`cloud-inference/models/asr/indicconformer/hindi/`

**Branch:** `feat/varshu-hindi-asr`

## 8.4 Varun — MLOps & Model Optimization Engineer

**Primary responsibility:** PellaV2 model quantization and deployment packaging.

Assigned work:

1. Inspect the original PellaV2 checkpoint.
2. Evaluate supported quantization approaches.
3. Generate optimized checkpoint variants.
4. Compare original and quantized model sizes.
5. Benchmark inference latency and resource consumption.
6. Validate detection performance after quantization.
7. Select and document the final checkpoint.
8. Generate SHA-256 checksums.
9. Prepare model-serving integration requirements.
10. Package the selected model for Docker deployment.
11. Document model versioning and distribution.

**Ownership directory:**

`cloud-inference/models/pellav2/`

**Branch:** `feat/varun-pellav2-optimization`

---

# 9. Team Development Workflow

All contributors must follow the same Git workflow.

## 9.1 Branching Strategy

| Branch | Purpose |
|---|---|
| `main` | Stable integrated code |
| `feat/sainadh-api-integration` | Gateway and integration |
| `feat/pallavi-telugu-asr` | Telugu ASR |
| `feat/varshu-hindi-asr` | Hindi ASR |
| `feat/varun-pellav2-optimization` | PellaV2 optimization |

Branch names can be adjusted to match existing repository conventions.

## 9.2 Pull Request Process

1. Create a feature branch from the latest `main`.
2. Implement only the assigned component.
3. Run component-level tests.
4. Validate Docker builds.
5. Document dependencies and configuration.
6. Commit meaningful changes.
7. Push the feature branch.
8. Create a pull request against `main`.
9. Request review.
10. Merge only after integration compatibility is confirmed.

## 9.3 Ownership Boundaries

- Contributors must not modify another contributor's owned model implementation without coordination.
- Shared contracts must be agreed upon before integration.
- API contract changes require System Architect approval.
- Docker Compose changes affecting multiple services require integration review.
- Model checkpoints must not be committed casually.
- No one should push experimental changes directly to `main`.

---

# 10. Testing and Quality Assurance

CFIC follows component-level testing and end-to-end integration testing.

## 10.1 Component Testing

| Component | Required Validation |
|---|---|
| API Gateway | Authentication, rate limits, validation and job handling |
| LID | Language detection and confidence |
| Telugu ASR | Telugu transcription |
| Hindi ASR | Hindi transcription |
| English ASR | English transcription |
| PellaV2 | Acoustic spoof detection |
| Gemma RIAS | Conversational risk assessment |
| Orchestrator | Routing, timeouts and service coordination |

## 10.2 End-to-End Testing

Test the complete pipeline using genuine audio samples.

1. Telugu audio.
2. Hindi audio.
3. English audio.
4. Unsupported language.
5. Low-confidence language detection.
6. Corrupted audio.
7. Empty audio.
8. Long audio requiring chunking.
9. PellaV2 unavailable.
10. Gemma RIAS unavailable.
11. LID unavailable.
12. Concurrent requests and queue saturation.

## 10.3 Privacy Validation

- No unnecessary transcript exposure in logs.
- No raw audio leakage.
- Secure temporary audio storage.
- Proper cleanup of temporary files.
- No API credentials in logs.
- Structured errors without sensitive payload exposure.

## 10.4 Existing Regression Baseline

The Gemma RIAS integration was previously validated with:

| Test Suite | Passed |
|---|---:|
| API Gateway | 47/47 |
| Cloud Inference Orchestrator | 13/13 |
| Open-Vakgyata LID | 5/5 |
| English Whisper ASR | 5/5 |
| Gemma RIAS | 17/17 |
| **Total** | **87/87** |

These are historical validation results, not a substitute for testing the newly integrated pipeline.

---

# 11. Documentation and Project Maintenance

The following documentation must be maintained:

| Document | Responsibility |
|---|---|
| System Architecture | Sainadh |
| API Documentation | Sainadh |
| LID Documentation | Sainadh |
| Telugu ASR Documentation | Pallavi |
| Hindi ASR Documentation | Varshu |
| PellaV2 Model Documentation | Varun |
| Gemma RIAS Documentation | Sainadh |
| Docker Deployment Guide | Sainadh |
| Integration Testing Report | Sainadh |

All documentation must reflect the actual implementation.

---

# 12. Future Development Roadmap

The current HackVersion focuses on a functional, modular, containerized architecture.

Future improvements may include:

1. Additional Indian language-specific ASR checkpoints.
2. Improved PellaV2 quantization and inference acceleration.
3. Distributed job queues.
4. Redis-backed rate limiting.
5. PostgreSQL-backed job persistence.
6. Prometheus monitoring and performance dashboards.
7. Horizontal scaling of inference services.
8. GPU-enabled inference deployment.
9. Advanced acoustic and semantic risk fusion.
10. Production-grade authentication and authorization.

These are future enhancements and must not be represented as already implemented.

---

# 13. Architecture Decisions and Explicit Exclusions

The following decisions define the current architecture.

| Area | Current Decision |
|---|---|
| Overall architecture | Decoupled hybrid microservices |
| API entry point | FastAPI API Gateway |
| Inference coordination | Cloud Inference Orchestrator |
| Language identification | Open-Vakgyata |
| Telugu ASR | IndicConformer Telugu |
| Hindi ASR | IndicConformer Hindi |
| English ASR | Whisper Base English |
| Acoustic spoof detection | PellaV2 |
| Conversational risk assessment | Gemma 4 E2B IT |
| Gemma format | GGUF Q4_K_M |
| Inference runtime | llama.cpp |
| Audio chunking | Five-second windows |
| Containerization | Docker Compose |
| Internal networking | Private Docker bridge |
| Model storage | Separate versioned model artifacts |

---

# 14. Project Completion Criteria

CFIC × VoiceShield HackVersion will be considered integration-ready when:

1. All required model services are independently containerized.
2. LID correctly identifies supported languages.
3. The orchestrator selects the correct ASR checkpoint.
4. Telugu, Hindi and English transcription are operational.
5. PellaV2 performs genuine acoustic inference.
6. Gemma RIAS performs genuine conversational inference.
7. The API Gateway handles authentication, rate limiting, chunking and job management.
8. The decision engine combines model outputs correctly.
9. All required services communicate through private Docker networking.
10. End-to-end audio testing succeeds.
11. Existing regression tests remain passing.
12. Documentation reflects the final implementation.
13. The repository has no obsolete VMC-South architecture dependencies.

---

# 15. Project Identity

**Project:** CloudForge Intelligence Core (CFIC) × VoiceShield HackVersion

**Team:** The SAILors

**Organization:** SAI Labs

**Event:** HackWithAditya 2027

**Repository Lead & System Architect:** Kasukurthi Yuvateja Sainadh

**Mission:**

Building intelligent, modular and scalable voice security infrastructure to protect individuals and organizations against AI-generated voice manipulation and conversational fraud.

**The SAILors — Building Intelligence. Securing Voices.**
