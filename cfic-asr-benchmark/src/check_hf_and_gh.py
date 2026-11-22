"""Check HF token and GitHub repo documentation."""

import os
import sys
from huggingface_hub import HfApi, get_token
import urllib.request
import json

token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or get_token()
print("Has HF Token in environment/cache:", bool(token))

if token:
    api = HfApi(token=token)
    try:
        info = api.model_info("ai4bharat/indic-conformer-600m-multilingual")
        print("Authenticated access to ai4bharat/indic-conformer-600m-multilingual SUCCESSFUL!")
    except Exception as e:
        print(f"Authenticated access error: {e}")
else:
    print("No HF_TOKEN found in environment or ~/.cache/huggingface/token.")

# Check official GitHub repo https://raw.githubusercontent.com/AI4Bharat/IndicConformerASR/main/README.md
try:
    req = urllib.request.Request(
        "https://raw.githubusercontent.com/AI4Bharat/IndicConformerASR/main/README.md",
        headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req) as response:
        gh_readme = response.read().decode("utf-8")
    print("\n=== Official IndicConformerASR GitHub README.md (first 3000 chars) ===")
    print(gh_readme[:3000])
except Exception as e:
    print(f"Error fetching GitHub README: {e}")
