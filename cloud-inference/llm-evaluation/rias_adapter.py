"""Temporary RIAS LLM inference adapter prototype.

Conforms to Cloud Inference architectural specifications:
- Accepts structured RIAS transcript requests.
- Evaluates transcript risk via local GGUF quantized model.
- Returns strict categorical verdicts: LOW, MEDIUM, or HIGH.
- Preserves session_id, chunk_index, timestamp_start_s, and timestamp_end_s.
- Fail-safe: empty/whitespace produces 'skipped'; errors/timeouts/malformed outputs
  produce 'error' with risk_level='INCONCLUSIVE', NEVER defaulting to LOW.
"""

import time
from typing import Optional
from pathlib import Path
from pydantic import BaseModel, Field
from llama_cpp import Llama

from prompts.rias_prompt import RIAS_SYSTEM_PROMPT, build_user_prompt, parse_rias_response


class RIASRequest(BaseModel):
    session_id: Optional[str] = Field(default=None, description="Unique caller session identifier")
    chunk_index: int = Field(default=0, description="Sequential audio chunk index")
    timestamp_start_s: float = Field(default=0.0, description="Chunk start timestamp in seconds")
    timestamp_end_s: float = Field(default=10.0, description="Chunk end timestamp in seconds")
    transcript: str = Field(..., description="ASR decoded transcript text")
    language: Optional[str] = Field(default=None, description="Detected ISO language code")


class RIASResponse(BaseModel):
    status: str = Field(default="success", description="'success', 'skipped', 'inconclusive', or 'error'")
    risk_level: str = Field(default="INCONCLUSIVE", description="'LOW', 'MEDIUM', 'HIGH', or 'INCONCLUSIVE'")
    reason: Optional[str] = Field(default=None, description="Concise rationale under 25 words")
    model_id: str = Field(..., description="Evaluated candidate model ID")
    model_revision: Optional[str] = Field(default=None, description="Git commit SHA of model weights")
    latency_ms: float = Field(default=0.0, description="Inference latency in milliseconds")
    session_id: Optional[str] = Field(default=None)
    chunk_index: int = Field(default=0)
    timestamp_start_s: float = Field(default=0.0)
    timestamp_end_s: float = Field(default=10.0)


class RIASAdapter:
    """In-memory inference adapter hosting the evaluated LLM candidate."""

    def __init__(
        self,
        model_path: str | Path,
        model_id: str = "Qwen/Qwen3.5-2B",
        model_revision: Optional[str] = "15852e8c16360a2fea060d615a32b45270f8a8fc",
        n_threads: int = 4,
        n_ctx: int = 2048
    ):
        self.model_path = Path(model_path)
        self.model_id = model_id
        self.model_revision = model_revision
        self.n_threads = n_threads
        self.n_ctx = n_ctx
        self._llm: Optional[Llama] = None

    def load_model(self) -> float:
        """Initialize Llama instance and return load time in seconds."""
        t0 = time.perf_counter()
        self._llm = Llama(
            model_path=str(self.model_path),
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            verbose=False
        )
        return round(time.perf_counter() - t0, 3)

    def assess(self, request: RIASRequest) -> RIASResponse:
        """Assess semantic fraud risk of a transcript request."""
        clean_text = (request.transcript or "").strip()

        # Empty / whitespace handling
        if not clean_text:
            return RIASResponse(
                status="skipped",
                risk_level="INCONCLUSIVE",
                reason="Empty or whitespace-only transcript",
                model_id=self.model_id,
                model_revision=self.model_revision,
                latency_ms=0.0,
                session_id=request.session_id,
                chunk_index=request.chunk_index,
                timestamp_start_s=request.timestamp_start_s,
                timestamp_end_s=request.timestamp_end_s
            )

        if self._llm is None:
            self.load_model()

        start_time = time.perf_counter()
        try:
            user_prompt = build_user_prompt(clean_text)
            messages = [
                {"role": "system", "content": RIAS_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]

            completion = self._llm.create_chat_completion(
                messages=messages,
                temperature=0.0,
                max_tokens=64,
                stop=["\n\n", "}"]
            )
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

            raw_text = completion["choices"][0]["message"]["content"] or ""
            if "{" in raw_text and not raw_text.strip().endswith("}"):
                raw_text = raw_text.strip() + "}"

            risk, reason, is_valid = parse_rias_response(raw_text)

            if not is_valid:
                return RIASResponse(
                    status="inconclusive",
                    risk_level="INCONCLUSIVE",
                    reason=f"LLM produced non-standard output: {raw_text[:60]}",
                    model_id=self.model_id,
                    model_revision=self.model_revision,
                    latency_ms=elapsed_ms,
                    session_id=request.session_id,
                    chunk_index=request.chunk_index,
                    timestamp_start_s=request.timestamp_start_s,
                    timestamp_end_s=request.timestamp_end_s
                )

            return RIASResponse(
                status="success",
                risk_level=risk,
                reason=reason,
                model_id=self.model_id,
                model_revision=self.model_revision,
                latency_ms=elapsed_ms,
                session_id=request.session_id,
                chunk_index=request.chunk_index,
                timestamp_start_s=request.timestamp_start_s,
                timestamp_end_s=request.timestamp_end_s
            )

        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return RIASResponse(
                status="error",
                risk_level="INCONCLUSIVE",
                reason=f"Inference exception: {type(exc).__name__}",
                model_id=self.model_id,
                model_revision=self.model_revision,
                latency_ms=elapsed_ms,
                session_id=request.session_id,
                chunk_index=request.chunk_index,
                timestamp_start_s=request.timestamp_start_s,
                timestamp_end_s=request.timestamp_end_s
            )
