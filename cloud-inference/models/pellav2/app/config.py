"""Configuration settings for PellaV2 K12 INT8 Acoustic Deepfake Detection Service."""

import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    service_name: str = "model-pellav2"
    port: int = 8321
    model_version: str = "pellav2-k12-int8"
    model_path: str = os.getenv("PELLAV2_MODEL_PATH", "weights/pellav2_k12_int8_final.pt")
    calibration_path: str = os.getenv("PELLAV2_CALIBRATION_PATH", "config/k12_calibration.json")
    checksum_path: str = os.getenv("PELLAV2_CHECKSUM_PATH", "checksums.txt")
    expected_sha256: str = "41c14cd6252fd3e94475b3277eb13b6f8fc463d5edf5b741b521f14bcfda0f98"
    threshold: float = float(os.getenv("PELLAV2_THRESHOLD", "0.50"))
    use_mock: bool = os.getenv("PELLAV2_USE_MOCK", "false").lower() == "true"
    device: str = os.getenv("PELLAV2_DEVICE", "cpu").lower()
    max_audio_bytes: int = 25 * 1024 * 1024  # 25 MiB upload limit
    sample_rate: int = 16000

    model_config = SettingsConfigDict(
        env_prefix="PELLAV2_",
        extra="ignore",
    )


settings = Settings()
