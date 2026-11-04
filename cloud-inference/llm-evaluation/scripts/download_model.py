"""Download helper for GGUF model files with resume support and progress logging."""

import os
import sys
import time
import requests
from pathlib import Path

def download_file(url: str, dest_path: Path, chunk_size: int = 10 * 1024 * 1024):
    """Download a file with HTTP Range resume support."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    existing_bytes = 0
    if temp_path.exists():
        existing_bytes = temp_path.stat().st_size
        print(f"Resuming download from byte {existing_bytes} ({existing_bytes / (1024*1024):.2f} MB)...")

    headers = {"User-Agent": "Mozilla/5.0"}
    if existing_bytes > 0:
        headers["Range"] = f"bytes={existing_bytes}-"

    resp = requests.get(url, headers=headers, stream=True, timeout=30)
    if resp.status_code not in (200, 206):
        raise RuntimeError(f"HTTP error {resp.status_code}: {resp.text[:200]}")

    total_bytes = existing_bytes
    if "Content-Range" in resp.headers:
        total_bytes = int(resp.headers["Content-Range"].split("/")[-1])
    elif "Content-Length" in resp.headers and resp.status_code == 200:
        total_bytes = int(resp.headers["Content-Length"])
    elif "Content-Length" in resp.headers and resp.status_code == 206:
        total_bytes = existing_bytes + int(resp.headers["Content-Length"])

    print(f"Total file size: {total_bytes / (1024*1024):.2f} MB")

    mode = "ab" if existing_bytes > 0 and resp.status_code == 206 else "wb"
    if mode == "wb":
        existing_bytes = 0

    downloaded = existing_bytes
    t_start = time.perf_counter()
    last_log = t_start

    with open(temp_path, mode) as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                now = time.perf_counter()
                if (now - last_log) >= 5.0 or downloaded == total_bytes:
                    elapsed = now - t_start
                    speed = ((downloaded - existing_bytes) / (1024 * 1024)) / elapsed if elapsed > 0 else 0
                    pct = (downloaded / total_bytes) * 100 if total_bytes > 0 else 0
                    print(f"  Downloaded: {downloaded / (1024*1024):.1f} / {total_bytes / (1024*1024):.1f} MB ({pct:.1f}%) | Speed: {speed:.2f} MB/s")
                    last_log = now

    temp_path.rename(dest_path)
    print(f"Successfully downloaded {dest_path.name} to {dest_path}!")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python download_model.py <repo_id> <filename>")
        sys.exit(1)

    repo = sys.argv[1]
    filename = sys.argv[2]
    out_file = Path("cloud-inference/llm-evaluation/models") / filename
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    print(f"Starting download: {url} -> {out_file}")
    download_file(url, out_file)
