"""Audio preprocessing and validation utilities for CFIC x VoiceShield ASR Benchmark."""

import os
import sys
import io
import math
import logging
import numpy as np
import soundfile as sf
from typing import Tuple, List, Optional, Union

logger = logging.getLogger(__name__)

class AudioProcessingError(Exception):
    """Exception raised for errors during audio loading or preprocessing."""
    pass

def setup_ffmpeg_path() -> Optional[str]:
    """Ensure ffmpeg is in PATH, leveraging imageio-ffmpeg or venv Scripts if available."""
    # Check if already in PATH
    ffmpeg_in_path = shutil_which("ffmpeg")
    if ffmpeg_in_path:
        return ffmpeg_in_path
    
    # Check venv Scripts folder
    venv_scripts = os.path.join(os.path.dirname(sys.executable))
    ffmpeg_in_venv = os.path.join(venv_scripts, "ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    if os.path.exists(ffmpeg_in_venv):
        if venv_scripts not in os.environ.get("PATH", ""):
            os.environ["PATH"] = venv_scripts + os.pathsep + os.environ.get("PATH", "")
        return ffmpeg_in_venv

    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        exe_dir = os.path.dirname(exe)
        # On Windows, imageio-ffmpeg binary might be named ffmpeg-win-x86_64-*.exe
        target_ffmpeg = os.path.join(venv_scripts, "ffmpeg.exe" if os.name == "nt" else "ffmpeg")
        if not os.path.exists(target_ffmpeg) and os.path.exists(exe):
            import shutil
            shutil.copyfile(exe, target_ffmpeg)
        if venv_scripts not in os.environ.get("PATH", ""):
            os.environ["PATH"] = venv_scripts + os.pathsep + os.environ.get("PATH", "")
        return target_ffmpeg
    except Exception as e:
        logger.warning(f"Could not configure imageio_ffmpeg: {e}")
        return None

def shutil_which(cmd: str) -> Optional[str]:
    import shutil
    return shutil.which(cmd)

def load_and_preprocess_audio(
    source: Union[str, bytes, io.BytesIO],
    target_sample_rate: Optional[int] = None,
    to_mono: bool = True
) -> Tuple[np.ndarray, int, int, float, List[str]]:
    """
    Safely load audio data, validate integrity, and apply necessary standard preprocessing.
    
    Returns:
        audio: 1D np.ndarray float32 normalized to [-1.0, 1.0]
        original_sr: int
        final_sr: int
        duration_seconds: float
        operations_applied: list of strings recording transformations
    """
    operations: List[str] = []
    
    try:
        if isinstance(source, bytes):
            file_obj = io.BytesIO(source)
            audio, sr = sf.read(file_obj, dtype="float32")
            operations.append("loaded_from_bytes")
        elif isinstance(source, io.BytesIO):
            source.seek(0)
            audio, sr = sf.read(source, dtype="float32")
            operations.append("loaded_from_bytesio")
        elif isinstance(source, str):
            if not os.path.exists(source):
                raise AudioProcessingError(f"Audio file does not exist: {source}")
            audio, sr = sf.read(source, dtype="float32")
            operations.append(f"loaded_from_file:{os.path.basename(source)}")
        else:
            raise AudioProcessingError(f"Unsupported source type: {type(source)}")
    except Exception as e:
        raise AudioProcessingError(f"Failed to decode audio: {str(e)}") from e

    original_sr = sr

    # Validate audio array
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
            operations.append(f"preserved_{channels}_channels")
    else:
        operations.append("already_mono")

    # Resampling if requested and different from source sample rate
    final_sr = original_sr
    if target_sample_rate is not None and target_sample_rate != original_sr:
        from scipy.signal import resample_poly
        gcd = math.gcd(original_sr, target_sample_rate)
        up = target_sample_rate // gcd
        down = original_sr // gcd
        audio = resample_poly(audio, up, down).astype(np.float32)
        final_sr = target_sample_rate
        operations.append(f"resampled_{original_sr}Hz_to_{target_sample_rate}Hz")
    else:
        operations.append(f"preserved_sample_rate_{original_sr}Hz")

    # Ensure float32
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    duration_seconds = float(len(audio) / final_sr)
    return audio, original_sr, final_sr, duration_seconds, operations

def save_wav_file(path: str, audio: np.ndarray, sample_rate: int) -> str:
    """Save float32 audio as 16-bit PCM WAV."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    # Clip to prevent overflow
    clipped = np.clip(audio, -1.0, 1.0)
    sf.write(path, clipped, sample_rate, subtype="PCM_16")
    return path
