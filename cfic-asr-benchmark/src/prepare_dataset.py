"""Dataset preparation and manifest generation for IndicSynth."""

import os
import sys
import json
import hashlib
import logging
from typing import Optional, Dict, Any, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def prepare_subset(
    dataset_name: str = "vdivyasharma/IndicSynth",
    config_name: str = "Telugu",
    language_code: str = "te",
    target_count: int = 100,
    output_dir: str = "data",
    data_file_shard: Optional[str] = None,
    hf_token: Optional[str] = None
) -> Dict[str, Any]:
    """
    Stream and materialize an evaluation subset of audio and transcripts.
    Saves:
    - Audio WAV files in {output_dir}/audio/{config_name.lower()}/
    - Manifest JSONL in {output_dir}/manifest_{config_name.lower()}.jsonl
    """
    from datasets import load_dataset, Audio
    import numpy as np
    from audio_utils import save_wav_file, load_and_preprocess_audio, AudioProcessingError

    audio_dir = os.path.join(output_dir, "audio", config_name.lower())
    os.makedirs(audio_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, f"manifest_{config_name.lower()}.jsonl")
    failed_log_path = os.path.join(output_dir, f"failed_{config_name.lower()}.jsonl")

    # If specific shard provided or default for Telugu
    if data_file_shard:
        data_file = data_file_shard
    elif config_name.lower() == "telugu":
        data_file = "Telugu/train-00128-of-00129.parquet"
    else:
        data_file = None

    if data_file:
        logger.info(f"Connecting to Hugging Face dataset: {dataset_name} (data_files={data_file}, streaming=True)...")
        ds = load_dataset(dataset_name, data_files=data_file, split="train", streaming=True, token=hf_token)
    else:
        logger.info(f"Connecting to Hugging Face dataset: {dataset_name} (config={config_name}, streaming=True)...")
        ds = load_dataset(dataset_name, config_name, split="train", streaming=True, token=hf_token)

    # Avoid torchcodec dependency by disabling automated dataset audio decoding
    ds = ds.cast_column("audio", Audio(decode=False))

    manifest_records: List[Dict[str, Any]] = []
    failed_records: List[Dict[str, Any]] = []
    
    record_idx = 0
    collected_count = 0

    for item in ds:
        if collected_count >= target_count:
            break
            
        sample_id = f"{config_name.lower()}_{collected_count:05d}"
        target_wav_path = os.path.join(audio_dir, f"{sample_id}.wav")

        try:
            audio_field = item.get("audio")
            raw_transcript = item.get("TTS Transcript")
            
            # Skip samples without TTS Transcript (such as voice-conversion samples)
            if not raw_transcript or not str(raw_transcript).strip():
                continue

            if not audio_field:
                raise AudioProcessingError("Missing audio field in dataset record")
            
            # Decode audio
            if isinstance(audio_field, dict):
                audio_bytes = audio_field.get("bytes")
                if audio_bytes is not None:
                    audio, orig_sr, final_sr, duration, ops = load_and_preprocess_audio(
                        audio_bytes, target_sample_rate=None, to_mono=True
                    )
                elif "array" in audio_field and audio_field["array"] is not None:
                    arr = np.array(audio_field["array"], dtype=np.float32)
                    audio, orig_sr, final_sr, duration, ops = load_and_preprocess_audio(
                        arr, target_sample_rate=None, to_mono=True
                    )
                else:
                    raise AudioProcessingError("Audio field dictionary has neither bytes nor array")
            else:
                raise AudioProcessingError(f"Unexpected audio field type: {type(audio_field)}")

            # Save clean WAV file
            save_wav_file(target_wav_path, audio, final_sr)

            # Compute sha256 of saved audio file for strict reproducibility
            with open(target_wav_path, "rb") as f:
                audio_sha256 = hashlib.sha256(f.read()).hexdigest()

            record = {
                "sample_id": sample_id,
                "dataset_index": record_idx,
                "language": config_name,
                "language_code": language_code,
                "audio_path": os.path.abspath(target_wav_path),
                "reference_transcript": str(raw_transcript).strip(),
                "generative_model": item.get("Generative Model", "unknown"),
                "source_speaker_id": str(item.get("Source Speaker_ID", "")),
                "target_speaker_id": str(item.get("Target Speaker ID", "")),
                "gender": item.get("Gender", "unknown"),
                "original_sample_rate": orig_sr,
                "sample_rate": final_sr,
                "audio_duration_seconds": round(duration, 4),
                "preprocessing_operations": ops,
                "audio_sha256": audio_sha256,
            }
            manifest_records.append(record)
            collected_count += 1
            if collected_count % 10 == 0 or collected_count == target_count:
                logger.info(f"Collected {collected_count}/{target_count} {config_name} samples...")

        except Exception as e:
            failed_info = {
                "sample_id": sample_id,
                "dataset_index": record_idx,
                "error": str(e),
            }
            failed_records.append(failed_info)
            logger.warning(f"Failed processing sample {record_idx}: {e}")

        record_idx += 1

    # Write manifest JSONL
    with open(manifest_path, "w", encoding="utf-8") as f:
        for r in manifest_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Write failed log if any
    if failed_records:
        with open(failed_log_path, "w", encoding="utf-8") as f:
            for r in failed_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(f"Manifest written to {manifest_path} ({len(manifest_records)} records, {len(failed_records)} failed)")
    
    return {
        "config_name": config_name,
        "manifest_path": manifest_path,
        "total_collected": len(manifest_records),
        "total_failed": len(failed_records),
        "target_count": target_count,
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="Telugu")
    parser.add_argument("--lang_code", default="te")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--output_dir", default="data")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    prepare_subset(
        config_name=args.config,
        language_code=args.lang_code,
        target_count=args.count,
        output_dir=args.output_dir,
        hf_token=token
    )
