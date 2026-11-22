"""Inspect IndicSynth dataset configurations and available languages."""

import sys
from datasets import get_dataset_config_names

try:
    print("Fetching dataset config names for 'vdivyasharma/IndicSynth'...")
    configs = get_dataset_config_names("vdivyasharma/IndicSynth")
    print(f"Total configs found: {len(configs)}")
    print("Configs:", configs)
except Exception as e:
    print(f"Error fetching configs: {e}")
