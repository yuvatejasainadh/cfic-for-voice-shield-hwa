"""Whisper inference runner for CFIC x VoiceShield ASR Benchmark."""

import os
import sys
import time
import json
import gc
import logging
import psutil
import pandas as pd
from typing import Dict, Any, List, Optional

from audio_utils import setup_ffmpeg_path, load_and_preprocess_audio
from chunking import create_audio_chunks, reconcile_chunk_transcripts
from evaluate import compute_corpus_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def load_manifest(manifest_path: str) -> List[Dict[str, Any]]:
    records = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))
    return records

def run_whisper_benchmark(
    manifest_path: str,
    model_name: str = "small",
    language_code: str = "te",
    language_name: str = "Telugu",
    device: str = "cpu",
    output_csv: str = "outputs/results/whisper_telugu.csv",
    chunk_output_csv: str = "outputs/results/chunk_results.csv",
    test_chunking: bool = True,
    chunk_duration_sec: float = 5.0,
    chunk_overlap_sec: float = 0.5,
    max_samples: Optional[int] = None
) -> Dict[str, Any]:
    """Run Whisper benchmark on manifest audio recordings."""
    import whisper

    setup_ffmpeg_path()
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(chunk_output_csv)), exist_ok=True)

    records = load_manifest(manifest_path)
    if max_samples:
        records = records[:max_samples]

    logger.info(f"Initializing Whisper '{model_name}' on {device}...")
    proc = psutil.Process()
    ram_before = proc.memory_info().rss / (1024 * 1024)

    t0 = time.perf_counter()
    model = whisper.load_model(model_name, device=device)
    init_time = time.perf_counter() - t0
    logger.info(f"Model initialized in {init_time:.2f}s")

    results = []
    chunk_results = []
    peak_ram = proc.memory_info().rss / (1024 * 1024)

    for idx, rec in enumerate(records):
        sample_id = rec["sample_id"]
        audio_path = rec["audio_path"]
        reference = rec["reference_transcript"]
        duration = rec.get("audio_duration_seconds", 0.0)

        # 1. Whole recording inference
        inf_start = time.perf_counter()
        prediction = ""
        success = False
        error_msg = ""
        
        try:
            # Transcription mode only, explicit language override
            res = model.transcribe(
                audio_path,
                language=language_code,
                task="transcribe",
                temperature=0.0,
                fp16=False # CPU requires fp16=False
            )
            prediction = res.get("text", "").strip()
            success = True
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error transcribing {sample_id}: {e}")

        inf_time = time.perf_counter() - inf_start
        rtf = (inf_time / duration) if duration > 0 else 0.0

        current_ram = proc.memory_info().rss / (1024 * 1024)
        if current_ram > peak_ram:
            peak_ram = current_ram

        results.append({
            "sample_id": sample_id,
            "language": language_name,
            "model_name": f"whisper_{model_name}",
            "reference": reference,
            "prediction": prediction,
            "audio_duration_seconds": round(duration, 4),
            "inference_seconds": round(inf_time, 4),
            "real_time_factor": round(rtf, 4),
            "success": success,
            "error": error_msg
        })

        # 2. Chunking benchmark (5s no-overlap and 5s with overlap) if enabled
        if test_chunking and success:
            try:
                audio_arr, orig_sr, final_sr, _, _ = load_and_preprocess_audio(audio_path, target_sample_rate=16000)
                
                # Strategies to test:
                # - 5s no-overlap (overlap=0.0)
                # - 5s with overlap (overlap=chunk_overlap_sec)
                strategies = [
                    ("5s_no_overlap", 0.0),
                    ("5s_with_overlap", chunk_overlap_sec)
                ]

                for strat_name, overlap in strategies:
                    chunks = create_audio_chunks(
                        audio_arr,
                        sample_rate=final_sr,
                        chunk_duration_sec=chunk_duration_sec,
                        overlap_sec=overlap
                    )
                    chunk_preds = []
                    total_chunk_time = 0.0

                    for c in chunks:
                        c_idx = c["chunk_index"]
                        c_dur = c["duration_sec"]
                        c_audio = c["audio"]

                        c_t0 = time.perf_counter()
                        c_pred = ""
                        c_ok = False
                        c_err = ""
                        try:
                            # Whisper can transcribe numpy array directly in 16kHz float32
                            c_res = model.transcribe(
                                c_audio,
                                language=language_code,
                                task="transcribe",
                                temperature=0.0,
                                fp16=False
                            )
                            c_pred = c_res.get("text", "").strip()
                            c_ok = True
                        except Exception as ce:
                            c_err = str(ce)

                        c_inf_time = time.perf_counter() - c_t0
                        c_rtf = (c_inf_time / c_dur) if c_dur > 0 else 0.0
                        total_chunk_time += c_inf_time
                        chunk_preds.append(c_pred)

                        chunk_results.append({
                            "sample_id": sample_id,
                            "model_name": f"whisper_{model_name}",
                            "chunk_strategy": strat_name,
                            "chunk_index": c_idx,
                            "start_sec": c["start_sec"],
                            "end_sec": c["end_sec"],
                            "chunk_duration_sec": c_dur,
                            "inference_seconds": round(c_inf_time, 4),
                            "real_time_factor": round(c_rtf, 4),
                            "raw_prediction": c_pred,
                            "success": c_ok,
                            "error": c_err
                        })

                    # Reconcile chunk texts
                    has_overlap = (overlap > 0.0)
                    recon = reconcile_chunk_transcripts(chunk_preds, has_overlap=has_overlap)
                    
                    # Record whole-reconciled comparison
                    results.append({
                        "sample_id": f"{sample_id}_{strat_name}",
                        "language": language_name,
                        "model_name": f"whisper_{model_name}_{strat_name}",
                        "reference": reference,
                        "prediction": recon["combined_text"],
                        "audio_duration_seconds": round(duration, 4),
                        "inference_seconds": round(total_chunk_time, 4),
                        "real_time_factor": round((total_chunk_time / duration) if duration > 0 else 0.0, 4),
                        "success": True,
                        "error": ""
                    })

            except Exception as e:
                logger.warning(f"Chunking benchmark error on {sample_id}: {e}")

        logger.info(
            f"[{idx+1}/{len(records)}] {sample_id} ({duration:.2f}s) - "
            f"RTF: {rtf:.3f} | Latency: {inf_time:.2f}s | Success: {success}"
        )
        sys.stdout.flush()

        # Incremental save to disk so no progress is ever lost
        pd.DataFrame(results).to_csv(output_csv, index=False, encoding="utf-8")
        if chunk_results:
            pd.DataFrame(chunk_results).to_csv(chunk_output_csv, index=False, encoding="utf-8")

    # Final output save
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv, index=False, encoding="utf-8")
    logger.info(f"Saved inference results to {output_csv}")

    if chunk_results:
        df_chunks = pd.DataFrame(chunk_results)
        # Append or write
        if os.path.exists(chunk_output_csv) and os.path.getsize(chunk_output_csv) > 0:
            df_chunks.to_csv(chunk_output_csv, mode="a", header=False, index=False, encoding="utf-8")
        else:
            df_chunks.to_csv(chunk_output_csv, index=False, encoding="utf-8")
        logger.info(f"Saved chunk results to {chunk_output_csv}")

    # Compute overall metrics for this model run
    # Extract only the base model runs (not the chunked variants) for standard metrics
    base_results = [r for r in results if not ("5s_no_overlap" in r["model_name"] or "5s_with_overlap" in r["model_name"])]
    metrics = compute_corpus_metrics(
        references=[r["reference"] for r in base_results],
        predictions=[r["prediction"] for r in base_results],
        durations=[r["audio_duration_seconds"] for r in base_results],
        latencies=[r["inference_seconds"] for r in base_results],
        success_flags=[r["success"] for r in base_results],
        model_init_time=init_time,
        peak_ram_mb=peak_ram
    )

    # Clean up model from memory
    del model
    gc.collect()

    return {
        "model_name": f"whisper_{model_name}",
        "language": language_name,
        "metrics": metrics,
        "results_count": len(base_results)
    }

if __name__ == "__main__":
    import argparse
    if sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifest_telugu.jsonl")
    parser.add_argument("--model", "--model-name", dest="model", default="small")
    parser.add_argument("--lang_code", "--lang-code", dest="lang_code", default="te")
    parser.add_argument("--lang_name", "--lang-name", dest="lang_name", default="Telugu")
    parser.add_argument("--output_csv", "--output", dest="output_csv", default="outputs/results/whisper_telugu.csv")
    parser.add_argument("--chunk_output_csv", "--chunk-output", dest="chunk_output_csv", default="outputs/results/chunk_results.csv")
    parser.add_argument("--max_samples", "--max-samples", dest="max_samples", type=int, default=None)
    args = parser.parse_args()

    metrics = run_whisper_benchmark(
        manifest_path=args.manifest,
        model_name=args.model,
        language_code=args.lang_code,
        language_name=args.lang_name,
        output_csv=args.output_csv,
        chunk_output_csv=args.chunk_output_csv,
        max_samples=args.max_samples
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
