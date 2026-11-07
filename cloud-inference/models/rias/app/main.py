"""Gemma 4 E2B IT RIAS LLM Inference Microservice.

Dedicated internal inference microservice for semantic risk assessment
within CloudForge Intelligence Core (CFIC) × VoiceShield.

Model: google/gemma-4-E2B-it
Quantization: GGUF Q4_K_M (unsloth/gemma-4-E2B-it-GGUF)
Runtime: llama.cpp via llama-cpp-python (CPU-first)
"""

import asyncio
from contextlib import asynccontextmanager
import hashlib
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import JSONResponse
import psutil
import sys

_app_dir = str(Path(__file__).resolve().parent)
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)
_parent_dir = str(Path(__file__).resolve().parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

try:
    from app.config import settings
    from app.prompt import RIAS_SYSTEM_PROMPT, build_user_prompt, parse_risk_label
    from app.schemas import (
        AssessRequest,
        AssessResponse,
        HealthResponse,
        MetadataResponse,
        ReadinessResponse,
    )
except ImportError:
    from config import settings
    from prompt import RIAS_SYSTEM_PROMPT, build_user_prompt, parse_risk_label
    from schemas import (
        AssessRequest,
        AssessResponse,
        HealthResponse,
        MetadataResponse,
        ReadinessResponse,
    )

logger = logging.getLogger("rias_gemma_service")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)

# Runtime state
_runtime: Dict[str, Any] = {
    "llm": None,
    "loaded": False,
    "device": "cpu",
    "error": None,
    "active_requests": 0,
    "waiting_requests": 0,
}

_semaphore: Optional[asyncio.Semaphore] = None
_lock = asyncio.Lock()


def verify_sha256(file_path: Path, expected_hash: str) -> bool:
    """Validate SHA-256 checksum of model file."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(10 * 1024 * 1024):
            sha.update(chunk)
    actual_hash = sha.hexdigest().lower()
    return actual_hash == expected_hash.lower()


def _resolve_model_path() -> Path:
    """Find model path from env, explicit settings, or fallback candidate path."""
    explicit = Path(settings.model_path)
    if explicit.exists():
        return explicit

    # Fallback to local evaluation model path if running locally
    eval_path = (
        Path(__file__).resolve().parents[3]
        / "llm-evaluation"
        / "models"
        / "gemma-4-E2B-it-Q4_K_M.gguf"
    )
    if eval_path.exists():
        return eval_path

    return explicit


def _load_model() -> None:
    """Load the locked Gemma GGUF model once at startup."""
    if settings.use_mock or os.getenv("RIAS_USE_MOCK", "false").lower() == "true":
        logger.info("RIAS service configured with USE_MOCK=true; skipping weight loading.")
        _runtime["loaded"] = True
        _runtime["device"] = "mock"
        return

    model_file = _resolve_model_path()
    if not model_file.exists():
        err = f"Model file not found at: {model_file}"
        logger.error(err)
        _runtime["loaded"] = False
        _runtime["error"] = err
        return

    if settings.require_checksum:
        logger.info(f"Verifying SHA-256 for {model_file.name}...")
        if not verify_sha256(model_file, settings.model_sha256):
            err = f"SHA-256 checksum mismatch for {model_file.name}"
            logger.error(err)
            _runtime["loaded"] = False
            _runtime["error"] = err
            return
        logger.info("SHA-256 checksum verified successfully.")

    try:
        from llama_cpp import Llama

        logger.info(
            f"Loading Gemma 4 E2B IT ({model_file.name}) via llama.cpp (threads={settings.n_threads}, n_ctx={settings.n_ctx})..."
        )
        t0 = time.perf_counter()
        llm = Llama(
            model_path=str(model_file),
            n_ctx=settings.n_ctx,
            n_threads=settings.n_threads,
            verbose=False,
        )
        elapsed = round(time.perf_counter() - t0, 3)

        process = psutil.Process(os.getpid())
        ram_mb = round(process.memory_info().rss / (1024 * 1024), 2)

        _runtime["llm"] = llm
        _runtime["loaded"] = True
        _runtime["device"] = "cpu"
        _runtime["error"] = None
        logger.info(
            f"Gemma 4 E2B IT successfully loaded in {elapsed}s. Process RSS: {ram_mb} MB"
        )
    except Exception as exc:
        err = f"Failed to load Gemma model: {type(exc).__name__}: {str(exc)}"
        logger.error(err)
        _runtime["loaded"] = False
        _runtime["error"] = err


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _semaphore
    _semaphore = asyncio.Semaphore(settings.max_concurrent_requests)
    _load_model()
    yield
    # Graceful shutdown
    if _runtime["llm"] is not None:
        logger.info("Unloading Gemma model during shutdown...")
        del _runtime["llm"]
        _runtime["llm"] = None
        _runtime["loaded"] = False


app = FastAPI(
    title="CFIC Gemma 4 E2B IT RIAS Inference Service",
    version="1.0.0",
    description="Internal containerized microservice serving Google Gemma 4 E2B IT Q4_K_M for RIAS semantic risk assessment",
    lifespan=lifespan,
)


async def verify_internal_auth(
    x_internal_token: Optional[str] = Header(None, alias="X-Internal-Token"),
    authorization: Optional[str] = Header(None),
):
    """Verify internal inter-service authentication token if configured."""
    expected = settings.internal_api_key
    if not expected:
        return True  # Internal network boundary security

    token = x_internal_token
    if not token and authorization and authorization.startswith("Bearer "):
        token = authorization[7:]

    if token != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized internal inference request",
        )
    return True


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Process and runtime health check."""
    loaded = _runtime["loaded"]
    is_mock = _runtime["device"] == "mock" or settings.use_mock
    return HealthResponse(
        status="ok" if (loaded or is_mock) else "degraded",
        service=settings.service_name,
        loaded=loaded or is_mock,
        model_id=settings.model_id,
        model_revision=settings.model_revision,
        device=_runtime["device"],
        is_mock=is_mock,
        error=_runtime.get("error"),
    )


@app.get("/ready", response_model=ReadinessResponse, tags=["System"])
async def ready():
    """Readiness endpoint verifying model is loaded and ready for inference."""
    is_mock = _runtime["device"] == "mock" or settings.use_mock
    if not (_runtime["loaded"] or is_mock):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Gemma model is not ready: {_runtime.get('error', 'Model not loaded')}",
        )
    return ReadinessResponse(
        status="ready",
        loaded=True,
        model_id=settings.model_id,
        active_requests=_runtime["active_requests"],
        max_concurrency=settings.max_concurrent_requests,
    )


@app.get("/metadata", response_model=MetadataResponse, tags=["System"])
async def metadata():
    """Service and model metadata."""
    return MetadataResponse(
        service=settings.service_name,
        model_id=settings.model_id,
        model_revision=settings.model_revision,
        quantization="Q4_K_M",
        architecture="Gemma4ForConditionalGeneration",
        runtime="llama.cpp",
        execution="CPU-first",
        n_ctx=settings.n_ctx,
        n_threads=settings.n_threads,
        allowed_labels=["LOW", "MEDIUM", "HIGH"],
    )


def _mock_inference(transcript: str) -> str:
    """Fast deterministic mock for CI/test environments."""
    lower = transcript.lower()
    high_keywords = [
        "otp",
        "password",
        "pin",
        "cvv",
        "arrest",
        "police",
        "cbi",
        "digital arrest",
        "transfer money",
        "wire funds",
        "anydesk",
        "teamviewer",
        "disconnect electricity",
        "customs",
        "narcotics",
    ]
    med_keywords = [
        "special offer",
        "call back",
        "pre-approved",
        "lottery",
        "prize",
        "urgent renewal",
        "unsolicited",
        "verify details",
    ]
    if any(k in lower for k in high_keywords):
        return "HIGH"
    if any(k in lower for k in med_keywords):
        return "MEDIUM"
    return "LOW"


def _execute_sync_inference(user_prompt: str) -> str:
    """Run synchronous llama.cpp chat completion."""
    llm = _runtime["llm"]
    if llm is None:
        raise RuntimeError("Model is not loaded")

    messages = [
        {"role": "system", "content": RIAS_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    completion = llm.create_chat_completion(
        messages=messages,
        temperature=0.0,
        max_tokens=settings.max_tokens,
        stop=["\n", "\n\n", "}", "<"],
    )
    return completion["choices"][0]["message"]["content"] or ""


@app.post(
    "/v1/assess",
    response_model=AssessResponse,
    tags=["Inference"],
    summary="Semantic fraud risk assessment of conversational audio transcript",
)
async def assess(
    req: AssessRequest,
    _auth: bool = Depends(verify_internal_auth),
):
    """Assess transcript risk and return strictly LOW, MEDIUM, or HIGH."""
    job_id = req.get_job_id()
    chunk_index = req.chunk_index
    t_start = time.perf_counter()

    # 1. Transcript validation
    clean_text = (req.transcript or "").strip()
    if not clean_text:
        return AssessResponse(
            job_id=job_id,
            chunk_index=chunk_index,
            status="SKIPPED",
            risk_level="INCONCLUSIVE",
            failure_reason="EMPTY_TRANSCRIPT",
            model_id=settings.model_id,
            model_revision=settings.model_revision,
            latency_ms=0.0,
        )

    if len(clean_text) > settings.max_transcript_length:
        elapsed = round((time.perf_counter() - t_start) * 1000, 2)
        return AssessResponse(
            job_id=job_id,
            chunk_index=chunk_index,
            status="INCONCLUSIVE",
            risk_level="INCONCLUSIVE",
            failure_reason="TRANSCRIPT_OVERSIZED",
            model_id=settings.model_id,
            model_revision=settings.model_revision,
            latency_ms=elapsed,
        )

    # 2. Check model readiness
    is_mock = _runtime["device"] == "mock" or settings.use_mock
    if not (_runtime["loaded"] or is_mock):
        elapsed = round((time.perf_counter() - t_start) * 1000, 2)
        return AssessResponse(
            job_id=job_id,
            chunk_index=chunk_index,
            status="INCONCLUSIVE",
            risk_level="INCONCLUSIVE",
            failure_reason=f"MODEL_UNAVAILABLE: {_runtime.get('error', 'Model not loaded')}",
            model_id=settings.model_id,
            model_revision=settings.model_revision,
            latency_ms=elapsed,
        )

    # 3. Queue & Concurrency Management
    if _runtime["waiting_requests"] >= settings.max_queue_size:
        elapsed = round((time.perf_counter() - t_start) * 1000, 2)
        logger.warning(f"RIAS queue full for job {job_id} chunk {chunk_index}")
        return AssessResponse(
            job_id=job_id,
            chunk_index=chunk_index,
            status="INCONCLUSIVE",
            risk_level="INCONCLUSIVE",
            failure_reason="QUEUE_OVERFLOW",
            model_id=settings.model_id,
            model_revision=settings.model_revision,
            latency_ms=elapsed,
        )

    _runtime["waiting_requests"] += 1
    acquired = False
    try:
        if _semaphore is None:
            raise RuntimeError("Concurrency semaphore not initialized")

        try:
            await asyncio.wait_for(
                _semaphore.acquire(), timeout=settings.queue_timeout_seconds
            )
            acquired = True
        except asyncio.TimeoutError:
            elapsed = round((time.perf_counter() - t_start) * 1000, 2)
            logger.warning(f"Queue wait timeout for job {job_id}")
            return AssessResponse(
                job_id=job_id,
                chunk_index=chunk_index,
                status="INCONCLUSIVE",
                risk_level="INCONCLUSIVE",
                failure_reason="QUEUE_TIMEOUT",
                model_id=settings.model_id,
                model_revision=settings.model_revision,
                latency_ms=elapsed,
            )

        _runtime["waiting_requests"] -= 1
        _runtime["active_requests"] += 1

        # 4. Inference Execution
        user_prompt = build_user_prompt(clean_text)

        if is_mock:
            raw_output = _mock_inference(clean_text)
        else:
            try:
                raw_output = await asyncio.wait_for(
                    asyncio.to_thread(_execute_sync_inference, user_prompt),
                    timeout=settings.timeout_seconds,
                )
            except asyncio.TimeoutError:
                elapsed = round((time.perf_counter() - t_start) * 1000, 2)
                logger.error(f"Inference timeout ({settings.timeout_seconds}s) for job {job_id}")
                return AssessResponse(
                    job_id=job_id,
                    chunk_index=chunk_index,
                    status="INCONCLUSIVE",
                    risk_level="INCONCLUSIVE",
                    failure_reason="INFERENCE_TIMEOUT",
                    model_id=settings.model_id,
                    model_revision=settings.model_revision,
                    latency_ms=elapsed,
                )

        elapsed = round((time.perf_counter() - t_start) * 1000, 2)

        # 5. Output Validation
        risk, is_valid, failure_reason = parse_risk_label(raw_output)

        if not is_valid:
            logger.warning(
                f"Job {job_id} chunk {chunk_index} invalid output: {failure_reason}"
            )
            return AssessResponse(
                job_id=job_id,
                chunk_index=chunk_index,
                status="INCONCLUSIVE",
                risk_level="INCONCLUSIVE",
                failure_reason=failure_reason,
                model_id=settings.model_id,
                model_revision=settings.model_revision,
                latency_ms=elapsed,
            )

        # Privacy invariant: never log raw transcript text
        logger.info(
            f"Assessed job={job_id} chunk={chunk_index} risk={risk} latency={elapsed}ms"
        )
        return AssessResponse(
            job_id=job_id,
            chunk_index=chunk_index,
            status="SUCCESS",
            risk_level=risk,
            failure_reason=None,
            model_id=settings.model_id,
            model_revision=settings.model_revision,
            latency_ms=elapsed,
        )

    except Exception as exc:
        elapsed = round((time.perf_counter() - t_start) * 1000, 2)
        logger.error(f"Inference exception for job {job_id}: {type(exc).__name__}: {str(exc)}")
        return AssessResponse(
            job_id=job_id,
            chunk_index=chunk_index,
            status="INCONCLUSIVE",
            risk_level="INCONCLUSIVE",
            failure_reason=f"RUNTIME_EXCEPTION: {type(exc).__name__}",
            model_id=settings.model_id,
            model_revision=settings.model_revision,
            latency_ms=elapsed,
        )
    finally:
        if acquired and _semaphore is not None:
            _semaphore.release()
            _runtime["active_requests"] = max(0, _runtime["active_requests"] - 1)
