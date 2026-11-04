"""Shared audio preprocessing and normalization utilities for Cloud Inference."""

import io
import math
import logging
from typing import Tuple, List, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)


class AudioProcessingError(Exception):
    """Exception raised for errors during audio loading or preprocessing."""
    pass


def load_and_preprocess_audio(
    source: Union[bytes, io.BytesIO],
    target_sample_rate: int = 16000,
    to_mono: bool = True,
) -> Tuple[np.ndarray, int, float, List[str]]:
    """Load audio bytes, validate integrity, and apply standard 16 kHz mono preprocessing.
    
    Returns:
        audio: 1D np.ndarray float32 normalized to [-1.0, 1.0]
        sample_rate: int (target sample rate)
        duration_seconds: float
        operations_applied: list of strings recording transformations
    """
    operations: List[str] = []

    try:
        import soundfile as sf
        if isinstance(source, bytes):
            source = io.BytesIO(source)
        source.seek(0)
        audio, sr = sf.read(source, dtype="float32")
        operations.append("loaded_via_soundfile")
    except Exception as exc:
        raise AudioProcessingError(f"Failed to decode audio: {exc}") from exc

    if audio is None or audio.size == 0:
        raise AudioProcessingError("Decoded audio is empty")

    if np.isnan(audio).any() or np.isinf(audio).any():
        raise AudioProcessingError("Decoded audio contains NaN or Inf values")

    # Multi-channel to mono conversion
    if audio.ndim > 1:
        channels = audio.shape[1] if audio.ndim == 2 else audio.shape[-1]
        if to_mono:
            audio = np.mean(audio, axis=-1)
            operations.append(f"converted_to_mono_from_{channels}_channels")
    else:
        operations.append("already_mono")

    # Resampling if needed
    if sr != target_sample_rate:
        try:
            from scipy.signal import resample_poly
            gcd = math.gcd(sr, target_sample_rate)
            up = target_sample_rate // gcd
            down = sr // gcd
            audio = resample_poly(audio, up, down).astype(np.float32)
            operations.append(f"resampled_{sr}Hz_to_{target_sample_rate}Hz")
        except ImportError:
            # Fallback linear interpolation
            num_samples = int(len(audio) * target_sample_rate / sr)
            audio = np.interp(
                np.linspace(0, len(audio), num_samples, endpoint=False),
                np.arange(len(audio)),
                audio,
            ).astype(np.float32)
            operations.append(f"interpolated_{sr}Hz_to_{target_sample_rate}Hz")
    else:
        operations.append(f"preserved_{target_sample_rate}Hz")

    # Ensure float32 normalized
    audio = np.clip(audio.astype(np.float32), -1.0, 1.0)
    duration_seconds = float(len(audio) / target_sample_rate)

    return audio, target_sample_rate, duration_seconds, operations
