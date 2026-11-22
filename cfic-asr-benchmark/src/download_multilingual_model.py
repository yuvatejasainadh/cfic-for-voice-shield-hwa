"""Download AI4Bharat IndicConformer 600M Multilingual NeMo checkpoint with resume support."""

import os
import sys
import time
import urllib.request
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("model_downloader")

MODEL_URL = "https://objectstore.e2enetworks.net/indicconformer/models/indicconformer_stt_multi_hybrid_rnnt_600m.nemo"
DEST_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "indicconformer_stt_multi_hybrid_rnnt_600m.nemo")

def download_file(url: str, dest_path: str):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req) as resp:
        total_size = int(resp.headers.get("Content-Length", 0))

    logger.info(f"Target file: {dest_path}")
    logger.info(f"Remote size: {total_size:,} bytes ({total_size / (1024**3):.2f} GB)")

    existing_size = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0

    if existing_size == total_size and total_size > 0:
        logger.info(f"File already completely downloaded ({existing_size:,} bytes). Skipping.")
        return dest_path

    headers = {"User-Agent": "Mozilla/5.0"}
    mode = "wb"
    if existing_size > 0 and existing_size < total_size:
        logger.info(f"Resuming download from byte {existing_size:,}...")
        headers["Range"] = f"bytes={existing_size}-"
        mode = "ab"

    req = urllib.request.Request(url, headers=headers)
    t0 = time.time()
    last_log_time = t0
    bytes_downloaded = existing_size

    with urllib.request.urlopen(req) as response, open(dest_path, mode) as out_file:
        chunk_size = 1024 * 1024 * 4  # 4 MB chunks
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            bytes_downloaded += len(chunk)

            now = time.time()
            if now - last_log_time >= 5.0 or bytes_downloaded == total_size:
                elapsed = now - t0
                speed = (bytes_downloaded - existing_size) / (elapsed + 1e-6) / (1024 * 1024)
                pct = (bytes_downloaded / total_size) * 100 if total_size > 0 else 0
                logger.info(f"Downloaded: {bytes_downloaded / (1024**3):.2f}/{total_size / (1024**3):.2f} GB ({pct:.1f}%) at {speed:.2f} MB/s")
                last_log_time = now

    logger.info(f"Download complete: {dest_path} ({os.path.getsize(dest_path):,} bytes)")
    return dest_path

if __name__ == "__main__":
    download_file(MODEL_URL, DEST_PATH)
