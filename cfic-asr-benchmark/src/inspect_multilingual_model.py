"""Inspect README.md, config.json, and model_onnx.py from Hugging Face."""

import sys
sys.stdout.reconfigure(encoding="utf-8")
from huggingface_hub import hf_hub_download

repo_id = "ai4bharat/indic-conformer-600m-multilingual"

for fname in ["README.md", "config.json", "model_onnx.py"]:
    try:
        path = hf_hub_download(repo_id=repo_id, filename=fname)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        print(f"=== {fname} ===")
        print(content[:3000])
        print("\n" + "="*40 + "\n")
    except Exception as e:
        print(f"Error fetching {fname}: {e}")
