"""Audio validation, ingestion, chunking, and storage service for API Gateway."""

import io
import os
import uuid
from typing import Any, BinaryIO, List
from fastapi import UploadFile
from app.core.config import settings
from app.core.errors import (
    AudioDurationTooLongException,
    AudioPayloadTooLargeException,
    CorruptedAudioPayloadException,
    EmptyAudioPayloadException,
    UnsupportedAudioFormatException,
)

ALLOWED_MIME_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/ogg",
    "audio/flac",
    "audio/x-flac",
    "application/octet-stream",
}

ALLOWED_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".ogg",
    ".flac",
}


class AudioService:
    """Manages audio payload validation, temporary storage, and chunking orchestration."""

    def __init__(self, max_bytes: int | None = None, storage_dir: str = "storage/audio"):
        self._max_bytes = max_bytes
        self.storage_dir = storage_dir

    @property
    def max_bytes(self) -> int:
        return self._max_bytes if self._max_bytes is not None else settings.max_audio_bytes

    @max_bytes.setter
    def max_bytes(self, value: int | None) -> None:
        self._max_bytes = value

    async def validate_upload(self, file: UploadFile) -> bytes:
        """Read and validate uploaded audio file.
        
        Raises EmptyAudioPayloadException or AudioPayloadTooLargeException.
        Returns the raw audio bytes.
        """
        content = await file.read()
        return self.validate_bytes(content)

    def validate_bytes(self, content: bytes) -> bytes:
        """Validate raw audio bytes against size constraints."""
        if not content:
            raise EmptyAudioPayloadException("Empty audio payload provided")

        actual_size = len(content)
        if actual_size > self.max_bytes:
            raise AudioPayloadTooLargeException(
                max_bytes=self.max_bytes,
                actual_bytes=actual_size,
            )

        return content

    def validate_audio_payload(
        self,
        content: bytes,
        filename: str = "audio.wav",
        content_type: str = "audio/wav",
    ) -> dict[str, Any]:
        """Validate audio payload MIME type, file extension, size limits, decodability, and duration cap.

        Raises:
            EmptyAudioPayloadException (400) if content is empty.
            AudioPayloadTooLargeException (413) if content exceeds max_audio_bytes.
            UnsupportedAudioFormatException (415) if MIME type or extension is unsupported.
            CorruptedAudioPayloadException (400) if audio bitstream cannot be decoded or has 0 duration.
            AudioDurationTooLongException (400) if audio duration exceeds max_audio_duration_seconds.

        Returns a dictionary with parsed audio metadata:
            duration_seconds, sample_rate, channels, size_bytes, mime_type, filename.
        """
        # 1. Empty payload check
        if not content:
            raise EmptyAudioPayloadException("Received empty audio payload (0 bytes)")

        # 2. Size limit check
        actual_size = len(content)
        if actual_size > self.max_bytes:
            raise AudioPayloadTooLargeException(
                max_bytes=self.max_bytes,
                actual_bytes=actual_size,
            )

        # 3. MIME type & extension checks
        clean_mime = (content_type or "").split(";")[0].strip().lower()
        ext = os.path.splitext(filename or "")[1].lower()

        if clean_mime not in ALLOWED_MIME_TYPES:
            raise UnsupportedAudioFormatException(
                f"Unsupported audio MIME type: '{clean_mime}'. Allowed: audio/wav, audio/mpeg, audio/ogg, audio/flac."
            )

        if ext and ext not in ALLOWED_EXTENSIONS:
            raise UnsupportedAudioFormatException(
                f"Unsupported audio file extension: '{ext}'. Allowed: .wav, .mp3, .ogg, .flac."
            )

        if clean_mime == "application/octet-stream" and ext not in ALLOWED_EXTENSIONS:
            raise UnsupportedAudioFormatException(
                "Binary payload (application/octet-stream) requires a supported audio file extension (.wav, .mp3, .ogg, .flac)."
            )

        # 4. Decodability & duration checks
        duration: float | None = None
        sample_rate: int | None = None
        channels: int | None = None

        # Fast header metadata inspect via soundfile
        try:
            import soundfile as sf
            with io.BytesIO(content) as bio:
                info = sf.info(bio)
                duration = float(info.duration)
                sample_rate = int(info.samplerate)
                channels = int(info.channels)
        except Exception:
            # Fallback to standard library wave module
            try:
                import wave
                with io.BytesIO(content) as bio:
                    with wave.open(bio, "rb") as wf:
                        sr = wf.getframerate()
                        frames = wf.getnframes()
                        channels = wf.getnchannels()
                        duration = float(frames / sr) if sr > 0 else 0.0
                        sample_rate = sr
            except Exception:
                duration = None

        if duration is None:
            raise CorruptedAudioPayloadException(
                "Audio payload is corrupted or contains unparseable bitstream"
            )

        if duration <= 0:
            raise CorruptedAudioPayloadException(
                "Audio stream contains zero duration or no audio samples"
            )

        max_duration = getattr(settings, "max_audio_duration_seconds", 120.0)
        if duration > max_duration:
            raise AudioDurationTooLongException(
                max_duration_s=max_duration,
                actual_duration_s=duration,
            )

        return {
            "duration_seconds": round(duration, 3),
            "sample_rate": sample_rate,
            "channels": channels,
            "size_bytes": actual_size,
            "mime_type": clean_mime,
            "filename": filename,
        }

    def store_audio_temporarily(self, audio_bytes: bytes, filename: str | None = None) -> str:
        """Save audio to a managed storage location for job processing."""
        os.makedirs(self.storage_dir, exist_ok=True)
        file_id = str(uuid.uuid4())
        ext = os.path.splitext(filename or "audio.wav")[1] or ".wav"
        dest_path = os.path.join(self.storage_dir, f"{file_id}{ext}")
        with open(dest_path, "wb") as f:
            f.write(audio_bytes)
        return dest_path

    def plan_5s_chunks(
        self, duration_seconds: float, window_seconds: float = 5.0, overlap_seconds: float = 0.5
    ) -> list[dict[str, float]]:
        """Calculate start and end time boundaries for 5-second windowed chunk orchestration."""
        chunks = []
        if duration_seconds <= 0:
            return chunks

        start = 0.0
        step = max(0.1, window_seconds - overlap_seconds)
        while start < duration_seconds:
            end = min(duration_seconds, start + window_seconds)
            chunks.append({
                "chunk_index": len(chunks),
                "start_seconds": round(start, 3),
                "end_seconds": round(end, 3),
                "duration_seconds": round(end - start, 3),
            })
            if end >= duration_seconds:
                break
            start += step
        return chunks

    def slice_into_5s_chunks(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        content_type: str = "audio/wav",
        chunk_duration_s: float = 5.0,
    ) -> List[dict[str, Any]]:
        """Slice audio bytes into sequential 5-second WAV chunks.

        If total duration <= chunk_duration_s, returns a single chunk.
        Falls back safely to raw bytes if audio decoding is unsupported or payload is mock.
        """
        # Try decoding with soundfile (supports WAV, MP3, FLAC, OGG)
        try:
            import soundfile as sf
            with io.BytesIO(audio_bytes) as bio:
                audio, sr = sf.read(bio, dtype="float32")
            if audio is not None and audio.size > 0:
                if audio.ndim > 1:
                    audio = audio.mean(axis=-1)
                total_duration = float(len(audio) / sr)

                if total_duration <= chunk_duration_s:
                    return [{
                        "chunk_index": 0,
                        "start_seconds": 0.0,
                        "end_seconds": round(total_duration, 3),
                        "audio_bytes": audio_bytes,
                        "filename": filename,
                        "content_type": content_type,
                    }]

                chunk_samples = int(chunk_duration_s * sr)
                chunks = []
                for i, start_idx in enumerate(range(0, len(audio), chunk_samples)):
                    chunk_audio = audio[start_idx : start_idx + chunk_samples]
                    buf = io.BytesIO()
                    sf.write(buf, chunk_audio, sr, format="WAV", subtype="PCM_16")
                    start_s = round(start_idx / sr, 3)
                    end_s = round((start_idx + len(chunk_audio)) / sr, 3)
                    chunks.append({
                        "chunk_index": i,
                        "start_seconds": start_s,
                        "end_seconds": end_s,
                        "audio_bytes": buf.getvalue(),
                        "filename": f"chunk_{i}.wav",
                        "content_type": "audio/wav",
                    })
                return chunks
        except Exception:
            pass

        # Try standard library wave module
        try:
            import wave
            with io.BytesIO(audio_bytes) as bio:
                with wave.open(bio, "rb") as wf:
                    sr = wf.getframerate()
                    n_frames = wf.getnframes()
                    n_channels = wf.getnchannels()
                    sampwidth = wf.getsampwidth()
                    total_duration = float(n_frames / sr)

                    if total_duration <= chunk_duration_s:
                        return [{
                            "chunk_index": 0,
                            "start_seconds": 0.0,
                            "end_seconds": round(total_duration, 3),
                            "audio_bytes": audio_bytes,
                            "filename": filename,
                            "content_type": content_type,
                        }]

                    chunk_frames = int(chunk_duration_s * sr)
                    chunks = []
                    for i, start_f in enumerate(range(0, n_frames, chunk_frames)):
                        wf.setpos(start_f)
                        frames = wf.readframes(chunk_frames)
                        buf = io.BytesIO()
                        with wave.open(buf, "wb") as out_wf:
                            out_wf.setnchannels(n_channels)
                            out_wf.setsampwidth(sampwidth)
                            out_wf.setframerate(sr)
                            out_wf.writeframes(frames)
                        start_s = round(start_f / sr, 3)
                        end_s = round(min(n_frames, start_f + chunk_frames) / sr, 3)
                        chunks.append({
                            "chunk_index": i,
                            "start_seconds": start_s,
                            "end_seconds": end_s,
                            "audio_bytes": buf.getvalue(),
                            "filename": f"chunk_{i}.wav",
                            "content_type": "audio/wav",
                        })
                    return chunks
        except Exception:
            pass

        # Fallback: single chunk with original bytes
        return [{
            "chunk_index": 0,
            "start_seconds": 0.0,
            "end_seconds": chunk_duration_s,
            "audio_bytes": audio_bytes,
            "filename": filename,
            "content_type": content_type,
        }]


audio_service = AudioService()
