"""Dataset inspection script for vdivyasharma/IndicSynth on Hugging Face."""

import os
import sys
import json
import logging
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def inspect_dataset_metadata(dataset_name: str = "vdivyasharma/IndicSynth", hf_token: Optional[str] = None) -> Dict[str, Any]:
    """Inspect dataset repository info, license, configs, and splits via huggingface_hub."""
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)
    
    logger.info(f"Inspecting dataset metadata for {dataset_name}...")
    info = api.dataset_info(dataset_name)
    
    card_data = info.card_data or {}
    configs = [c.get("config_name") for c in card_data.get("configs", [])] if "configs" in card_data else []
    
    report = {
        "dataset_id": info.id,
        "author": info.author,
        "private": info.private,
        "gated": info.gated,
        "license": card_data.get("license", "unknown"),
        "languages": card_data.get("language", []),
        "task_categories": card_data.get("task_categories", []),
        "available_configs": configs,
        "tags": info.tags
    }
    return report

def inspect_sample_example(
    dataset_name: str = "vdivyasharma/IndicSynth",
    config_name: str = "Telugu",
    split: str = "train",
    hf_token: Optional[str] = None
) -> Dict[str, Any]:
    """Inspect the first sample via streaming to verify schema, audio representation, and fields."""
    from datasets import load_dataset, Audio
    import io
    import soundfile as sf
    
    logger.info(f"Streaming 1 sample from {dataset_name} (config={config_name}, split={split})...")
    
    ds = load_dataset(dataset_name, config_name, split=split, streaming=True, token=hf_token)
    # Disable automatic audio decoding in datasets to use robust local soundfile decoding
    ds = ds.cast_column("audio", Audio(decode=False))
    sample = next(iter(ds))
    
    # Inspect fields
    fields = list(sample.keys())
    audio_field = sample.get("audio")
    
    audio_info = {}
    if isinstance(audio_field, dict):
        audio_info["keys"] = list(audio_field.keys())
        audio_info["path"] = audio_field.get("path")
        audio_bytes = audio_field.get("bytes")
        if audio_bytes is not None:
            audio_info["bytes_length"] = len(audio_bytes)
            # Test soundfile decoding
            data, sr = sf.read(io.BytesIO(audio_bytes))
            audio_info["decoded_shape"] = list(data.shape)
            audio_info["sampling_rate"] = sr
            audio_info["duration_sec"] = round(len(data) / sr, 4)
            audio_info["dtype"] = str(data.dtype)
    
    sample_report = {
        "fields": fields,
        "audio_info": audio_info,
        "generative_model": sample.get("Generative Model"),
        "source_speaker_id": sample.get("Source Speaker_ID"),
        "target_speaker_id": sample.get("Target Speaker ID"),
        "gender": sample.get("Gender"),
        "tts_transcript": sample.get("TTS Transcript"),
    }
    return sample_report

if __name__ == "__main__":
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    meta = inspect_dataset_metadata(hf_token=token)
    print("=== DATASET METADATA ===")
    print(json.dumps(meta, indent=2))
    
    if "Telugu" in meta["available_configs"]:
        sample_rep = inspect_sample_example(config_name="Telugu", hf_token=token)
        print("\n=== TELUGU SAMPLE INSPECTION ===")
        print(json.dumps(sample_rep, indent=2, ensure_ascii=False))
    else:
        print(f"Telugu not found in configs: {meta['available_configs']}")
