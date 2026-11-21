"""Configuration settings for Original PellaV2 Acoustic Deepfake Detection Service."""

import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    service_name: str = "model-pellav2"
    port: int = 8321
    model_version: str = "pellav2-original"
    architecture: str = "PellaV2-w2v-bert-2.0"
    model_path: str = os.getenv("PELLAV2_MODEL_PATH", "weights/pellav2_detector.pt")
    config_path: str = os.getenv("PELLAV2_CONFIG_PATH", "config/w2v_bert_config.json")
    checksum_path: str = os.getenv("PELLAV2_CHECKSUM_PATH", "checksums.txt")
    expected_sha256: str = "d21aedfb52e2f87706c2753715a29665482b0f4a034800c31579df5d458559ac"
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
