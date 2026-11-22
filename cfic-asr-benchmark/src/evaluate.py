"""Evaluation metrics and text normalization module for CFIC x VoiceShield ASR Benchmark."""

import re
import unicodedata
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import jiwer

# Standard punctuation characters to remove while preserving Indic diacritics
PUNCTUATION_REGEX = re.compile(r'[\.,!?;:\"\'\(\)\[\]\{\}\/\\—\-_~`@#\$%\^&\*\+=<>|।॥]')
WHITESPACE_REGEX = re.compile(r'\s+')

def normalize_indic_text(text: Optional[str]) -> str:
    """
    Standard text normalization for Indian languages (specifically Telugu) and general ASR:
    1. Handle None / empty inputs safely.
    2. Normalize Unicode to NFC form (canonical decomposition followed by canonical composition).
    3. Remove punctuation and special symbols (including Indian danda । and double danda ॥).
    4. Preserve Telugu characters, matras, virama, anusvara, and digits intact.
    5. Collapse multiple whitespace characters into a single space and strip leading/trailing spaces.
    
    Explicitly NOT doing:
    - No English transliteration.
    - No script alteration or phonetic smoothing.
    - No selective substitutions.
    """
    if not text:
        return ""
    
    # Unicode NFC normalization
    norm = unicodedata.normalize("NFC", str(text))
    
    # Strip punctuation
    norm = PUNCTUATION_REGEX.sub(" ", norm)
    
    # Collapse whitespaces
    norm = WHITESPACE_REGEX.sub(" ", norm).strip()
    
    return norm

def compute_corpus_metrics(
    references: List[str],
    predictions: List[str],
    durations: List[float],
    latencies: List[float],
    success_flags: List[bool],
    model_init_time: float = 0.0,
    peak_ram_mb: float = 0.0,
    peak_gpu_vram_mb: float = 0.0
) -> Dict[str, Any]:
    """
    Compute comprehensive corpus-level and sample-level metrics.
    
    Distinguishes clearly between:
    - Corpus WER: sum(word edit distances) / sum(reference word counts)
    - Mean clip WER: average of per-clip WERs
    - Corpus CER: sum(char edit distances) / sum(reference char counts)
    - Mean clip CER: average of per-clip CERs
    - Real-Time Factor (RTF): total inference time / total audio duration
    """
    total_samples = len(references)
    successful_samples = sum(1 for s in success_flags if s)
    failed_samples = total_samples - successful_samples
    success_rate = (successful_samples / total_samples * 100.0) if total_samples > 0 else 0.0

    # Filter to successful runs for transcription accuracy & latency metrics
    valid_refs = []
    valid_preds = []
    valid_durations = []
    valid_latencies = []
    
    for ref, pred, dur, lat, ok in zip(references, predictions, durations, latencies, success_flags):
        if ok:
            norm_ref = normalize_indic_text(ref)
            norm_pred = normalize_indic_text(pred)
            valid_refs.append(norm_ref)
            valid_preds.append(norm_pred)
            valid_durations.append(dur)
            valid_latencies.append(lat)

    if not valid_refs:
        return {
            "total_samples": total_samples,
            "successful_samples": successful_samples,
            "failed_samples": failed_samples,
            "success_rate_percent": success_rate,
            "model_init_seconds": model_init_time,
            "error": "No successful samples to evaluate"
        }

    # jiwer process_words computes total substitutions, deletions, insertions, hits
    word_output = jiwer.process_words(valid_refs, valid_preds)
    corpus_wer = word_output.wer
    
    char_output = jiwer.process_characters(valid_refs, valid_preds)
    corpus_cer = char_output.cer

    # Per-clip WER & CER
    clip_wers = []
    clip_cers = []
    clip_rtfs = []
    suspicious_pairs = []

    for idx, (ref, pred, dur, lat) in enumerate(zip(valid_refs, valid_preds, valid_durations, valid_latencies)):
        # Handle zero-length reference
        ref_words = ref.split()
        if len(ref_words) == 0:
            clip_wer = 0.0 if len(pred.split()) == 0 else 1.0
        else:
            clip_wer = jiwer.wer(ref, pred)
        clip_wers.append(clip_wer)

        if len(ref) == 0:
            clip_cer = 0.0 if len(pred) == 0 else 1.0
        else:
            clip_cer = jiwer.cer(ref, pred)
        clip_cers.append(clip_cer)

        rtf = lat / dur if dur > 0 else 0.0
        clip_rtfs.append(rtf)

        # Flag suspicious reference/prediction pairs for manual review
        # e.g., WER > 0.8, empty prediction on non-empty reference, extreme length ratio
        is_suspicious = False
        reasons = []
        if clip_wer > 0.85:
            is_suspicious = True
            reasons.append(f"high_wer_{clip_wer:.2f}")
        if len(pred) == 0 and len(ref) > 0:
            is_suspicious = True
            reasons.append("empty_prediction")
        if len(ref) > 0:
            len_ratio = len(pred) / len(ref)
            if len_ratio < 0.25 or len_ratio > 3.0:
                is_suspicious = True
                reasons.append(f"extreme_length_ratio_{len_ratio:.2f}")
        
        if is_suspicious:
            suspicious_pairs.append({
                "sample_idx": idx,
                "reference": ref,
                "prediction": pred,
                "wer": round(clip_wer, 4),
                "cer": round(clip_cer, 4),
                "reasons": ";".join(reasons)
            })

    total_audio_duration = sum(valid_durations)
    total_inference_time = sum(valid_latencies)
    corpus_rtf = (total_inference_time / total_audio_duration) if total_audio_duration > 0 else 0.0

    lat_array = np.array(valid_latencies)
    rtf_array = np.array(clip_rtfs)

    metrics = {
        "total_samples": total_samples,
        "successful_samples": successful_samples,
        "failed_samples": failed_samples,
        "success_rate_percent": round(success_rate, 2),
        "model_init_seconds": round(model_init_time, 4),
        "corpus_wer": round(corpus_wer, 4),
        "corpus_cer": round(corpus_cer, 4),
        "mean_clip_wer": round(float(np.mean(clip_wers)), 4),
        "median_clip_wer": round(float(np.median(clip_wers)), 4),
        "mean_clip_cer": round(float(np.mean(clip_cers)), 4),
        "median_clip_cer": round(float(np.median(clip_cers)), 4),
        "total_audio_duration_seconds": round(total_audio_duration, 2),
        "total_inference_seconds": round(total_inference_time, 2),
        "corpus_rtf": round(corpus_rtf, 4),
        "mean_clip_rtf": round(float(np.mean(rtf_array)), 4),
        "median_clip_rtf": round(float(np.median(rtf_array)), 4),
        "mean_latency_seconds": round(float(np.mean(lat_array)), 4),
        "median_latency_seconds": round(float(np.median(lat_array)), 4),
        "p95_latency_seconds": round(float(np.percentile(lat_array, 95)), 4) if len(lat_array) >= 20 else round(float(np.max(lat_array)), 4),
        "min_latency_seconds": round(float(np.min(lat_array)), 4),
        "max_latency_seconds": round(float(np.max(lat_array)), 4),
        "peak_ram_mb": round(peak_ram_mb, 2),
        "peak_gpu_vram_mb": round(peak_gpu_vram_mb, 2),
        "suspicious_pairs_count": len(suspicious_pairs),
        "suspicious_pairs": suspicious_pairs[:10]  # Store top 10 in summary
    }

    return metrics
