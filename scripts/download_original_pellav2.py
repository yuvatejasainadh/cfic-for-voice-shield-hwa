"""Download and verify original PellaV2 audio deepfake detector weights.

Repository: Sadanie/pellav2-audio-deepfake-detector
Revision: 94b899915ee551fa90d8844961532e4aaab58322
Artifact: pellav2_detector.pt
Expected SHA-256: d21aedfb52e2f87706c2753715a29665482b0f4a034800c31579df5d458559ac
"""

import hashlib
import os
import shutil
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

HUB_REPO = "Sadanie/pellav2-audio-deepfake-detector"
REVISION = "94b899915ee551fa90d8844961532e4aaab58322"
FILENAME = "pellav2_detector.pt"
EXPECTED_SHA256 = "d21aedfb52e2f87706c2753715a29665482b0f4a034800c31579df5d458559ac"

TARGET_DIR = Path(__file__).resolve().parent.parent / "cloud-inference" / "models" / "pellav2" / "weights"
TARGET_PATH = TARGET_DIR / FILENAME

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest().lower()

def download_and_verify():
    print(f"Target directory: {TARGET_DIR}")
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    if TARGET_PATH.exists():
        print(f"Found existing {TARGET_PATH}, verifying checksum...")
        actual_sha = compute_sha256(TARGET_PATH)
        if actual_sha == EXPECTED_SHA256.lower():
            print(f"Checksum verified: {actual_sha}")
            return str(TARGET_PATH)
        else:
            print(f"Checksum mismatch: expected {EXPECTED_SHA256}, got {actual_sha}. Re-downloading...")

    print(f"Downloading {FILENAME} from {HUB_REPO} (revision: {REVISION})...")
    downloaded_cache = hf_hub_download(
        repo_id=HUB_REPO,
        filename=FILENAME,
        revision=REVISION,
    )
    print(f"Downloaded to HF cache: {downloaded_cache}")

    print(f"Copying to target path: {TARGET_PATH}...")
    shutil.copy2(downloaded_cache, TARGET_PATH)

    print("Verifying downloaded file SHA-256...")
    actual_sha = compute_sha256(TARGET_PATH)
    if actual_sha != EXPECTED_SHA256.lower():
        raise ValueError(f"Checksum mismatch! Expected {EXPECTED_SHA256}, got {actual_sha}")

    print(f"SUCCESS: Checksum matches expected hash: {actual_sha}")
    print(f"File size: {TARGET_PATH.stat().st_size} bytes")

    # Also pre-cache w2v-bert-2.0 backbone
    print("\nPre-caching facebook/w2v-bert-2.0 backbone...")
    from transformers import Wav2Vec2BertModel
    backbone = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")
    print(f"Backbone successfully loaded: {type(backbone).__name__}")

    return str(TARGET_PATH)

if __name__ == "__main__":
    try:
        download_and_verify()
    except Exception as exc:
        print(f"Error downloading original model: {exc}", file=sys.stderr)
        sys.exit(1)
