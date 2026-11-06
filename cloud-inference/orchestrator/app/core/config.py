"""Configuration settings for Cloud Inference Orchestrator."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    service_name: str = "cloud-inference-orchestrator"
    port: int = 8200

    # Downstream Model Service URLs
    lid_service_url: str = "http://localhost:8301"
    asr_en_service_url: str = "http://localhost:8311"
    asr_te_service_url: str = "http://localhost:8312"
    asr_hi_service_url: str = "http://localhost:8313"
    vmc_south_service_url: str = "http://localhost:8321"
    aasist_fallback_service_url: str = "http://localhost:8322"
    rias_service_url: str = "http://localhost:8331"

    # Timeouts & Policies
    model_timeout_seconds: float = 15.0
    lid_timeout_seconds: float = 10.0
    asr_timeout_seconds: float = 15.0
    rias_timeout_seconds: float = 15.0

    # Policy controls
    lid_max_retries: int = 2
    lid_confidence_threshold: float = 0.60

    model_config = SettingsConfigDict(
        env_prefix="INFERENCE_",
        extra="ignore",
    )


settings = Settings()
