"""AI4Bharat IndicConformer inference runner for CFIC x VoiceShield ASR Benchmark."""

import os
import sys
import time
import json
import gc
import logging
import psutil
import tempfile
import soundfile as sf
import pandas as pd
from typing import Dict, Any, List, Optional

sys.stdout.reconfigure(encoding="utf-8")

from audio_utils import setup_ffmpeg_path, load_and_preprocess_audio
from chunking import create_audio_chunks, reconcile_chunk_transcripts

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def load_manifest(manifest_path: str) -> List[Dict[str, Any]]:
    records = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))
    return records

def transcribe_audio_file(model, audio_path: str, language_id: str = "te") -> str:
    """Helper to transcribe a single audio file with IndicConformer."""
    # Transcribe returns tuple or list of transcripts
    out = model.transcribe([audio_path], language_id=language_id)
    if isinstance(out, tuple):
        first_elem = out[0]
        if isinstance(first_elem, list) and len(first_elem) > 0:
            return str(first_elem[0]).strip()
        return str(first_elem).strip()
    elif isinstance(out, list) and len(out) > 0:
        return str(out[0]).strip()
    return str(out).strip()

def run_indicconformer_benchmark(
    manifest_path: str,
    nemo_model_path: str,
    model_name: str = "indicconformer_te_large",
    language_id: str = "te",
    language_name: str = "Telugu",
    decoder_type: str = "ctc",
    output_csv: str = "outputs/results/indicconformer_telugu.csv",
    chunk_output_csv: str = "outputs/results/chunk_results.csv",
    test_chunking: bool = True,
    chunk_duration_sec: float = 5.0,
    chunk_overlap_sec: float = 0.5,
    max_samples: Optional[int] = None
) -> Dict[str, Any]:
    """Run IndicConformer benchmark across manifest recordings."""
    import torch
    import nemo.collections.asr as nemo_asr

    setup_ffmpeg_path()
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(chunk_output_csv)), exist_ok=True)

    records = load_manifest(manifest_path)
    if max_samples:
        records = records[:max_samples]

    logger.info(f"Loading IndicConformer model '{model_name}' from {nemo_model_path} on CPU...")
    proc = psutil.Process()
    ram_before = proc.memory_info().rss / (1024 * 1024)

    t0 = time.perf_counter()
    model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(
        restore_path=nemo_model_path,
        map_location=torch.device("cpu")
    )
    if decoder_type in ["rnnt", "ctc"]:
        model.change_decoding_strategy(decoder_type=decoder_type)
    model.eval()
    init_time = time.perf_counter() - t0
    logger.info(f"Model initialized in {init_time:.2f}s with decoder: {getattr(model, 'cur_decoder', decoder_type)}")

    results = []
    chunk_results = []
    peak_ram = proc.memory_info().rss / (1024 * 1024)

    # Use a scratch directory for temporary audio chunks
    temp_chunk_dir = tempfile.mkdtemp(prefix="indic_chunks_")

    try:
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
                prediction = transcribe_audio_file(model, audio_path, language_id=language_id)
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
                "model_name": model_name,
                "reference": reference,
                "prediction": prediction,
                "audio_duration_seconds": round(duration, 3),
                "inference_seconds": round(inf_time, 4),
                "real_time_factor": round(rtf, 4),
                "success": success,
                "error": error_msg
            })

            logger.info(
                f"[{idx+1}/{len(records)}] {sample_id} ({duration:.2f}s) - "
                f"RTF: {rtf:.3f} | Latency: {inf_time:.2f}s | Success: {success}"
            )

            # 2. Chunking evaluation if requested
            if test_chunking and success:
                # Load preprocessed 16kHz audio for chunking
                audio_arr, orig_sr, final_sr, _, _ = load_and_preprocess_audio(audio_path, target_sample_rate=16000)

                for overlap in [0.0, chunk_overlap_sec]:
                    strategy_name = "5s_no_overlap" if overlap == 0.0 else "5s_with_overlap"
                    chunks = create_audio_chunks(audio_arr, sample_rate=final_sr, chunk_duration_sec=chunk_duration_sec, overlap_sec=overlap)

                    chunk_preds = []
                    chunk_latencies = []
                    chunk_failures = 0
                    c_start = time.perf_counter()

                    for c_idx, c_info in enumerate(chunks):
                        # Write chunk to temp WAV
                        tmp_wav = os.path.join(temp_chunk_dir, f"{sample_id}_{strategy_name}_{c_idx}.wav")
                        sf.write(tmp_wav, c_info["audio"], final_sr)

                        c_inf_start = time.perf_counter()
                        c_text = ""
                        try:
                            c_text = transcribe_audio_file(model, tmp_wav, language_id=language_id)
                        except Exception as ce:
                            chunk_failures += 1
                            logger.warning(f"Chunk failure {sample_id} chunk {c_idx}: {ce}")
                        c_inf_time = time.perf_counter() - c_inf_start
                        chunk_latencies.append(c_inf_time)
                        chunk_preds.append(c_text)

                        # Record granular chunk result
                        chunk_results.append({
                            "sample_id": sample_id,
                            "model_name": model_name,
                            "chunk_strategy": strategy_name,
                            "chunk_index": c_idx,
                            "start_sec": c_info["start_sec"],
                            "end_sec": c_info["end_sec"],
                            "chunk_duration_sec": c_info["duration_sec"],
                            "inference_seconds": round(c_inf_time, 4),
                            "real_time_factor": round((c_inf_time / c_info["duration_sec"]) if c_info["duration_sec"] > 0 else 0.0, 4),
                            "raw_prediction": c_text,
                            "success": c_text != "",
                            "error": ""
                        })

                        # Clean up chunk wav
                        try:
                            os.remove(tmp_wav)
                        except OSError:
                            pass

                    total_c_time = time.perf_counter() - c_start
                    c_rtf = (total_c_time / duration) if duration > 0 else 0.0

                    recon = reconcile_chunk_transcripts(chunk_preds, has_overlap=(overlap > 0))

                    results.append({
                        "sample_id": f"{sample_id}_{strategy_name}",
                        "language": language_name,
                        "model_name": f"{model_name}_{strategy_name}",
                        "reference": reference,
                        "prediction": recon["combined_text"],
                        "audio_duration_seconds": round(duration, 4),
                        "inference_seconds": round(total_c_time, 4),
                        "real_time_factor": round((total_c_time / duration) if duration > 0 else 0.0, 4),
                        "success": chunk_failures == 0,
                        "error": ""
                    })

            # Incremental save so results are immediately persisted
            pd.DataFrame(results).to_csv(output_csv, index=False, encoding="utf-8")
            sys.stdout.flush()

    finally:
        # Clean temp directory
        try:
            import shutil
            shutil.rmtree(temp_chunk_dir, ignore_errors=True)
        except Exception:
            pass

    # Save outputs
    df_res = pd.DataFrame(results)
    df_res.to_csv(output_csv, index=False, encoding="utf-8")
    logger.info(f"Saved primary results to {output_csv}")

    if chunk_results:
        df_chunk = pd.DataFrame(chunk_results)
        if os.path.exists(chunk_output_csv):
            existing = pd.read_csv(chunk_output_csv)
            # Remove any previous entries for this model
            existing = existing[existing["model_name"] != model_name]
            combined = pd.concat([existing, df_chunk], ignore_index=True)
            combined.to_csv(chunk_output_csv, index=False, encoding="utf-8")
        else:
            df_chunk.to_csv(chunk_output_csv, index=False, encoding="utf-8")
        logger.info(f"Saved chunking results to {chunk_output_csv}")

    # Explicit memory cleanup
    del model
    gc.collect()

    return {
        "model_name": model_name,
        "init_time_seconds": round(init_time, 3),
        "peak_ram_mb": round(peak_ram, 1),
        "num_samples": len(results),
        "successful_samples": sum(1 for r in results if r["success"]),
        "failed_samples": sum(1 for r in results if not r["success"])
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run IndicConformer ASR benchmark")
    parser.add_argument("--manifest", default="data/manifest_telugu.jsonl", help="Path to manifest")
    parser.add_argument("--nemo-model", default="models/indicconformer_stt_te_hybrid_rnnt_large.nemo", help="Path to .nemo model")
    parser.add_argument("--model-name", default="indicconformer_te_large", help="Model name identifier")
    parser.add_argument("--decoder", default="ctc", choices=["ctc", "rnnt"], help="Decoder strategy")
    parser.add_argument("--output", default="outputs/results/indicconformer_telugu.csv", help="Output CSV")
    parser.add_argument("--chunk-output", default="outputs/results/chunk_results.csv", help="Chunk output CSV")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to evaluate")
    parser.add_argument("--no-chunking", action="store_true", help="Disable chunking evaluation")

    args = parser.parse_args()
    summary = run_indicconformer_benchmark(
        manifest_path=args.manifest,
        nemo_model_path=args.nemo_model,
        model_name=args.model_name,
        decoder_type=args.decoder,
        output_csv=args.output,
        chunk_output_csv=args.chunk_output,
        test_chunking=not args.no_chunking,
        max_samples=args.max_samples
    )
    print("Benchmark completed:", json.dumps(summary, indent=2))
