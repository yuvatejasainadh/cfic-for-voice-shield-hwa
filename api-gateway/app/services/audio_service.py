"""Audio validation, ingestion, chunking, and storage service for API Gateway."""

import os
import uuid
from typing import BinaryIO
from fastapi import UploadFile
from app.core.config import settings
from app.core.errors import AudioPayloadTooLargeException, EmptyAudioPayloadException

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


class AudioService:
    """Manages audio payload validation, temporary storage, and chunking orchestration."""

    def __init__(self, max_bytes: int | None = None, storage_dir: str = "storage/audio"):
        self.max_bytes = max_bytes if max_bytes is not None else settings.max_audio_bytes
        self.storage_dir = storage_dir

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

    def store_audio_temporarily(self, audio_bytes: bytes, filename: str | None = None) -> str:
        """Save audio to a managed storage location for job processing."""
        os.makedirs(self.storage_dir, exist_ok=True)
        file_id = str(uuid.uuid4())
        ext = os.path.splitext(filename or "audio.wav")[1] or ".wav"
        dest_path = os.path.join(self.storage_dir, f"{file_id}{ext}")
        with open(dest_path, "wb") as f:
            f.write(audio_bytes)
        return dest_path

    def plan_5s_chunks(self, duration_seconds: float, window_seconds: float = 5.0, overlap_seconds: float = 0.5) -> list[dict[str, float]]:
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


audio_service = AudioService()
