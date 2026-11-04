"""Configuration settings for CFIC API Gateway."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "CFIC-VoiceShield-API-Gateway"
    environment: str = "development"
    api_key: str = "replace-me-for-local-testing"
    rate_limit_requests: int = 30
    rate_limit_window_seconds: int = 60
    max_audio_bytes: int = 25 * 1024 * 1024  # 25 MiB upload limit

    # Cloud Inference Subsystem URL (internal orchestration)
    cloud_inference_url: str = "http://cloud-inference-orchestrator:8200"
    cloud_inference_timeout: float = 30.0

    # Downstream legacy / fallback service URLs
    pella_base_url: str = "http://localhost:8101"
    transcription_base_url: str = "http://localhost:8102"
    askdrip_llm_base_url: str = ""
    askdrip_llm_api_key: str = ""
    askdrip_llm_model: str = "askdrip-risk-v1"
    request_timeout_seconds: float = 30.0

    model_config = SettingsConfigDict(
        env_prefix="CFIC_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
