"""Prepare evaluation recordings for 12 Indian languages for IndicConformer benchmark."""

import os
import sys
import json
import logging
import soundfile as sf
import numpy as np
from typing import Dict, Any, List

sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_multilingual")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
AUDIO_BASE_DIR = os.path.join(DATA_DIR, "audio")
MANIFEST_OUT = os.path.join(DATA_DIR, "manifest_multilingual_12lang.jsonl")

LANGUAGE_SPECS = {
    "te": {"name": "Telugu", "dataset": "vdivyasharma/IndicSynth", "config": "Telugu", "license": "CC BY-NC 4.0"},
    "ta": {"name": "Tamil", "dataset": "vdivyasharma/IndicSynth", "config": "Tamil", "license": "CC BY-NC 4.0"},
    "kn": {"name": "Kannada", "dataset": "vdivyasharma/IndicSynth", "config": "Kannada", "license": "CC BY-NC 4.0"},
    "ml": {"name": "Malayalam", "dataset": "vdivyasharma/IndicSynth", "config": "Malayalam", "license": "CC BY-NC 4.0"},
    "hi": {"name": "Hindi", "dataset": "vdivyasharma/IndicSynth", "config": "Hindi", "license": "CC BY-NC 4.0"},
    "bn": {"name": "Bengali", "dataset": "vdivyasharma/IndicSynth", "config": "Bengali", "license": "CC BY-NC 4.0"},
    "mr": {"name": "Marathi", "dataset": "vdivyasharma/IndicSynth", "config": "Marathi", "license": "CC BY-NC 4.0"},
    "gu": {"name": "Gujarati", "dataset": "vdivyasharma/IndicSynth", "config": "Gujarati", "license": "CC BY-NC 4.0"},
    "pa": {"name": "Punjabi", "dataset": "vdivyasharma/IndicSynth", "config": "Punjabi", "license": "CC BY-NC 4.0"},
    "or": {"name": "Odia", "dataset": "vdivyasharma/IndicSynth", "config": "Odia", "license": "CC BY-NC 4.0"},
    "ur": {"name": "Urdu", "dataset": "vdivyasharma/IndicSynth", "config": "Urdu", "license": "CC BY-NC 4.0"},
    "as": {"name": "Assamese", "dataset": "google/fleurs", "config": "as_in", "license": "CC BY 4.0"},
}

def load_existing_telugu_manifest(manifest_path: str) -> List[Dict[str, Any]]:
    records = []
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    item["language_code"] = "te"
                    item["dataset_source"] = "vdivyasharma/IndicSynth"
                    item["license"] = "CC BY-NC 4.0"
                    records.append(item)
    return records

def extract_dataset_subset(lang_code: str, spec: Dict[str, Any], target_count: int = 100) -> List[Dict[str, Any]]:
    from datasets import load_dataset
    lang_name = spec["name"]
    dataset_name = spec["dataset"]
    config = spec["config"]
    license_type = spec["license"]

    out_dir = os.path.join(AUDIO_BASE_DIR, lang_code)
    os.makedirs(out_dir, exist_ok=True)

    records = []
    logger.info(f"Streaming {target_count} samples for {lang_name} ({lang_code}) from {dataset_name} [{config}]...")

    try:
        split_name = "test" if dataset_name == "google/fleurs" else "train"
        ds = load_dataset(dataset_name, config, split=split_name, streaming=True)
    except Exception as e:
        logger.warning(f"Could not load split '{split_name}' for {lang_code}: {e}. Retrying with 'train'...")
        ds = load_dataset(dataset_name, config, split="train", streaming=True)

    count = 0
    for idx, item in enumerate(ds):
        if count >= target_count:
            break

        sample_id = f"{lang_code}_{count:05d}"
        wav_path = os.path.join(out_dir, f"{sample_id}.wav")

        try:
            # Handle audio column
            audio_field = item.get("audio", {})
            audio_arr = audio_field.get("array")
            orig_sr = audio_field.get("sampling_rate", 16000)

            if audio_arr is None or len(audio_arr) == 0:
                continue

            # Convert to float32
            if not isinstance(audio_arr, np.ndarray):
                audio_arr = np.array(audio_arr, dtype=np.float32)
            else:
                audio_arr = audio_arr.astype(np.float32)

            # Mono
            if audio_arr.ndim > 1:
                audio_arr = np.mean(audio_arr, axis=1)

            # Resample to 16000 if needed
            target_sr = 16000
            if orig_sr != target_sr:
                import librosa
                audio_arr = librosa.resample(audio_arr, orig_sr=orig_sr, target_sr=target_sr)

            duration = len(audio_arr) / target_sr
            if duration < 0.5:
                continue

            # Reference transcript
            ref = item.get("transcription") or item.get("transcript") or item.get("text") or item.get("raw_transcription") or ""
            ref = str(ref).strip()
            if not ref:
                continue

            # Save WAV
            sf.write(wav_path, audio_arr, target_sr, subtype="PCM_16")

            record = {
                "sample_id": sample_id,
                "language": lang_name,
                "language_code": lang_code,
                "audio_path": os.path.abspath(wav_path),
                "reference_transcript": ref,
                "audio_duration_seconds": round(duration, 3),
                "sample_rate": target_sr,
                "dataset_source": dataset_name,
                "dataset_config": config,
                "license": license_type
            }
            records.append(record)
            count += 1

            if count % 20 == 0:
                logger.info(f"Extracted {count}/{target_count} samples for {lang_name} ({lang_code})")

        except Exception as ex:
            logger.warning(f"Error processing record {idx} for {lang_code}: {ex}")

    logger.info(f"Finished {lang_name} ({lang_code}): {len(records)} valid samples saved.")
    return records

def main():
    target_samples_per_lang = 100
    all_records = []

    # 1. Reuse verified Telugu baseline
    telugu_manifest = os.path.join(DATA_DIR, "manifest_telugu.jsonl")
    te_records = load_existing_telugu_manifest(telugu_manifest)
    if te_records:
        logger.info(f"Reusing {len(te_records)} verified Telugu recordings from {telugu_manifest}")
        all_records.extend(te_records[:target_samples_per_lang])
    else:
        logger.warning("No existing Telugu manifest found; downloading fresh.")
        all_records.extend(extract_dataset_subset("te", LANGUAGE_SPECS["te"], target_samples_per_lang))

    # 2. Extract remaining 11 languages
    for code, spec in LANGUAGE_SPECS.items():
        if code == "te" and len(te_records) >= target_samples_per_lang:
            continue
        try:
            records = extract_dataset_subset(code, spec, target_count=target_samples_per_lang)
            all_records.extend(records)
        except Exception as e:
            logger.error(f"Failed to prepare dataset for {spec['name']} ({code}): {e}")

    # Write unified multilingual manifest
    with open(MANIFEST_OUT, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(f"Saved total {len(all_records)} multilingual evaluation records to {MANIFEST_OUT}")

if __name__ == "__main__":
    main()
