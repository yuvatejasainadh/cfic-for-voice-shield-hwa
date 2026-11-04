# CFIC × VoiceShield — Security & Authentication Architecture

## 1. Authentication Layer

- **Mechanism:** HTTP Header-based API Key (`X-API-Key`).
- **Gateway Enforcement:** Implemented in `api-gateway/app/core/security.py` and `api-gateway/app/services/auth_service.py`.
- **Validation:** Every inbound request to `/v1/analyze` must provide a key matching `CFIC_API_KEY`.
- **Unauthorized Failure:** Requests without a valid key receive HTTP `401 Unauthorized` with `WWW-Authenticate: ApiKey`.

---

## 2. Sliding-Window Rate Limiting

- **Implementation:** `api-gateway/app/core/rate_limiter.py`.
- **Key Derivation:** Client IP combined with API key hash suffix (`{client_ip}:{key[-6:]}`).
- **Thresholds:** Default 30 requests per 60-second window (configurable via `CFIC_RATE_LIMIT_REQUESTS` and `CFIC_RATE_LIMIT_WINDOW_SECONDS`).
- **Quota Exceeded:** Returns HTTP `429 Too Many Requests` with a standard `Retry-After` header indicating seconds until earliest window reset.
- **Clustered Scaling:** Infrastructure supports Redis-backed sliding logs via `infrastructure/redis/redis.conf`.

---

## 3. Network Isolation & Service Boundaries

- **Public Attack Surface:** Strictly limited to `api-gateway` port `8000`.
- **Model Isolation:** All model microservices (`model-lid`, `model-asr-*`, `model-vmc-south`, `model-aasist-fallback`, and `cloud-inference-orchestrator`) are attached exclusively to `cfic-internal-network` and do **not** expose host ports by default.
- **Zero Secrets in Git:** Secrets are configured strictly through environment variables (`.env`).
