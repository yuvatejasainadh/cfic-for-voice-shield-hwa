"""Inspect deepfake dataset structure and audio properties across all acoustic scenarios."""

import os
import json
from pathlib import Path
import soundfile as sf

DATASET_ROOT = Path(r"C:\Users\yuvat\Music\Deepfake Dataset")

def inspect_dataset():
    if not DATASET_ROOT.exists():
        print(f"Dataset root {DATASET_ROOT} does not exist.")
        return

    top_level_dirs = [d for d in DATASET_ROOT.iterdir() if d.is_dir()]
    print(f"Found {len(top_level_dirs)} top-level categories in {DATASET_ROOT}:")

    summary = {
        "dataset_root": str(DATASET_ROOT),
        "total_files": 0,
        "categories": {},
    }

    total_files = 0

    for cat_dir in sorted(top_level_dirs):
        gt_label = "FAKE" if "fake_voice" in cat_dir.name else "GENUINE"
        subdirs = [s for s in cat_dir.iterdir() if s.is_dir()]
        
        cat_info = {
            "ground_truth": gt_label,
            "subcategories": {},
            "total_files": 0,
        }

        for sub in sorted(subdirs):
            wav_files = list(sub.glob("*.wav"))
            count = len(wav_files)
            cat_info["total_files"] += count
            total_files += count

            sample_props = {}
            if wav_files:
                sample_file = wav_files[0]
                try:
                    info = sf.info(str(sample_file))
                    sample_props = {
                        "samplerate": info.samplerate,
                        "channels": info.channels,
                        "duration_sec": round(info.duration, 3),
                        "format": info.format,
                        "subtype": info.subtype,
                        "file_size_bytes": sample_file.stat().st_size,
                    }
                except Exception as e:
                    sample_props = {"error": str(e)}

            cat_info["subcategories"][sub.name] = {
                "count": count,
                "sample_properties": sample_props,
            }
            print(f"  [{gt_label}] {cat_dir.name}/{sub.name}: {count} wavs, sample={sample_props.get('samplerate')}Hz, {sample_props.get('duration_sec')}s")

        print(f"Total for {cat_dir.name}: {cat_info['total_files']} files")
        summary["categories"][cat_dir.name] = cat_info

    summary["total_files"] = total_files
    print(f"\nOverall dataset total: {total_files} files")

    output_path = Path("benchmark/deepfake_dataset_manifest.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Manifest written to {output_path}")

if __name__ == "__main__":
    inspect_dataset()
