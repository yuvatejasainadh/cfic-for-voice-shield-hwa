"""Inspect availability of Assamese in open datasets."""

from datasets import get_dataset_config_names

datasets_to_check = [
    "google/fleurs",
    "ai4bharat/IndicVoices",
    "vdivyasharma/IndicSynth"
]

for ds in datasets_to_check:
    try:
        cfgs = get_dataset_config_names(ds)
        as_matches = [c for c in cfgs if "as" in c.lower() or "assam" in c.lower()]
        print(f"{ds}: Found {len(cfgs)} configs. Assamese matches: {as_matches[:5]}")
    except Exception as e:
        print(f"Error checking {ds}: {e}")
