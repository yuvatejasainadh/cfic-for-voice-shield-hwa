"""Configuration settings for Gemma RIAS LLM Microservice."""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    service_name: str = "model-rias-gemma"
    port: int = 8331

    # Model specifications (Locked: Google Gemma 4 E2B IT Q4_K_M)
    model_id: str = "google/gemma-4-E2B-it"
    model_revision: str = "4adf15748de9e1a0097b6274cee9c15836c41c37"
    model_path: str = os.getenv(
        "GEMMA_MODEL_PATH",
        str(
            Path(__file__).resolve().parent.parent / "weights" / "gemma-4-E2B-it-Q4_K_M.gguf"
        ),
    )
    model_sha256: str = "740185b21d22ceb83a11c3aa62ad5842ef32c70f6096d756bbee85a1e4ec34b8"
    require_checksum: bool = False

    # Runtime and Hardware
    use_mock: bool = False
    n_threads: int = 4
    n_ctx: int = 2048
    max_tokens: int = 16
    timeout_seconds: float = 15.0
    queue_timeout_seconds: float = 5.0
    max_concurrent_requests: int = 2
    max_queue_size: int = 10
    max_transcript_length: int = 5000

    # Optional internal security token for inter-service validation
    internal_api_key: Optional[str] = None

    model_config = SettingsConfigDict(
        env_prefix="RIAS_",
        extra="ignore",
    )


settings = Settings()
