# CFIC × VoiceShield Local ASR Benchmark

A standalone, reproducible local ASR benchmarking framework designed for evaluating Automatic Speech Recognition candidates on synthetic and natural South Indian speech (Telugu), as an experimental supporting component for **CloudForge Intelligence Core (CFIC) × VoiceShield**.

## Directory Structure

```text
cfic-asr-benchmark/
├── README.md
├── config.yaml
├── requirements-whisper.txt
├── requirements-indicconformer.txt
├── .gitignore
├── src/
│   ├── inspect_dataset.py       # Selective download and schema inspection for IndicSynth
│   ├── prepare_dataset.py       # Audio extraction, resampling (16kHz), and manifest generation
│   ├── run_whisper.py           # Whisper runner (small & large-v3-turbo) with chunking
│   ├── run_indicconformer.py    # AI4Bharat NeMo IndicConformer runner with chunking
│   ├── audio_utils.py           # Audio I/O, format checks, ffmpeg path management
│   ├── chunking.py              # 5s windowing with overlap & longest-common-affix boundary reconciliation
│   ├── evaluate.py              # jiwer WER/CER with Indic text normalization & suspicious pair detection
│   ├── compile_summary.py       # Unified metric aggregator and report table generator
│   └── system_info.py           # Host hardware profiler (CPU, RAM, GPU, OS)
├── scripts/
│   ├── setup.ps1 / setup.sh
│   ├── run_whisper.ps1 / run_whisper.sh
│   └── run_indicconformer.ps1 / run_indicconformer.sh
├── data/
│   ├── manifest_telugu.jsonl    # Evaluation subset manifest (100 samples)
│   └── telugu/                  # Local 16kHz mono WAV audio clips
├── outputs/
│   ├── results/
│   │   ├── benchmark_summary.csv        # Consolidated comparative metrics
│   │   ├── indicconformer_100_telugu.csv # Full 100-sample IndicConformer results
│   │   ├── indicconformer_telugu.csv     # 20-sample validation & chunking results
│   │   ├── whisper_telugu.csv            # Whisper small comparative results
│   │   ├── whisper_turbo_telugu.csv      # Whisper large-v3-turbo comparative results
│   │   └── chunk_results.csv             # Granular chunk-level breakdown & tail analysis
│   └── reports/
│       └── benchmark_summary.md          # Comprehensive executive & technical benchmark report
└── tests/
    ├── test_audio.py
    ├── test_chunking.py
    └── test_evaluate.py
```

## Setup & Environments

To prevent dependency collisions between NeMo, PyTorch Lightning, Transformers, and Whisper, two isolated virtual environments are used:

### 1. Whisper Environment (`.venv-whisper`)
```powershell
python -m venv .venv-whisper
.\.venv-whisper\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install openai-whisper jiwer soundfile imageio-ffmpeg pandas pyyaml
```

### 2. IndicConformer Environment (`.venv-indic`)
```powershell
python -m venv .venv-indic
.\.venv-indic\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install Cython sentencepiece soundfile librosa pandas pyyaml transformers==4.38.0 pyannote.metrics pyannote.core
# AI4Bharat NeMo clone and install:
git clone -b v2 https://github.com/AI4Bharat/NeMo.git models/NeMo
pip install -e models/NeMo
```

## Running the Tests

```powershell
.\.venv-whisper\Scripts\python.exe -m unittest discover -s tests
```

## Executing the Benchmark

### 1. Dataset Preparation
```powershell
.\.venv-whisper\Scripts\python.exe src/prepare_dataset.py --dataset vdivyasharma/IndicSynth --language Telugu --samples 100
```

### 2. Run IndicConformer
```powershell
.\.venv-indic\Scripts\python.exe src/run_indicconformer.py --manifest data/manifest_telugu.jsonl --nemo-model models/indicconformer_stt_te_hybrid_rnnt_large.nemo --max-samples 20
```

### 3. Run Whisper
```powershell
.\.venv-whisper\Scripts\python.exe src/run_whisper.py --model small --manifest data/manifest_telugu.jsonl --max-samples 5
.\.venv-whisper\Scripts\python.exe src/run_whisper.py --model large-v3-turbo --manifest data/manifest_telugu.jsonl --max-samples 5
```

### 4. Compile Metrics
```powershell
.\.venv-whisper\Scripts\python.exe src/compile_summary.py
```

## Benchmark Findings Summary

- **IndicConformer Telugu CTC (`indicconformer_stt_te_hybrid_rnnt_large`)** is **12.5x to 36x faster than Whisper** on CPU (RTF **0.0802** vs 2.9178).
- **IndicConformer achieves 4.14% CER** on the matched Telugu evaluation subset (5.59% on 100 clips).
- **Whisper suffers from catastrophic repetition hallucination loops on 5-second chunked inputs** due to 30-second silence padding in its autoregressive decoder.
- **VoiceShield Architecture Recommendation:** Preserve the core detection pipeline (`LID -> AASIST-L Detector -> Shared Result`). Use IndicConformer strictly as an **asynchronous worker queue** for forensic transcript logging.
