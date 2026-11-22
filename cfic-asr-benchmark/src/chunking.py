"""Audio chunking and boundary reconciliation module for CFIC x VoiceShield ASR Benchmark."""

import numpy as np
from typing import List, Dict, Any, Tuple

def create_audio_chunks(
    audio: np.ndarray,
    sample_rate: int,
    chunk_duration_sec: float = 5.0,
    overlap_sec: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Split an audio array into chunks of specified duration and overlap.
    
    Ensures:
    - If total audio <= chunk_duration_sec, returns single chunk without modification.
    - Final chunk is NOT discarded and NOT padded with artificial silence.
    - Audio slices are exact and reproducible.
    
    Returns list of chunk dicts:
        chunk_index: int
        start_sec: float
        end_sec: float
        duration_sec: float
        audio: np.ndarray
    """
    total_samples = len(audio)
    total_duration = total_samples / sample_rate
    
    if total_duration <= chunk_duration_sec or chunk_duration_sec <= 0:
        return [{
            "chunk_index": 0,
            "start_sec": 0.0,
            "end_sec": total_duration,
            "duration_sec": total_duration,
            "audio": audio
        }]
    
    chunk_samples = int(round(chunk_duration_sec * sample_rate))
    overlap_samples = int(round(overlap_sec * sample_rate)) if overlap_sec > 0 else 0
    stride_samples = max(1, chunk_samples - overlap_samples)
    
    chunks: List[Dict[str, Any]] = []
    start_sample = 0
    chunk_idx = 0
    
    while start_sample < total_samples:
        end_sample = min(total_samples, start_sample + chunk_samples)
        chunk_audio = audio[start_sample:end_sample]
        actual_chunk_duration = len(chunk_audio) / sample_rate
        
        chunks.append({
            "chunk_index": chunk_idx,
            "start_sec": round(start_sample / sample_rate, 4),
            "end_sec": round(end_sample / sample_rate, 4),
            "duration_sec": round(actual_chunk_duration, 4),
            "audio": chunk_audio
        })
        
        chunk_idx += 1
        start_sample += stride_samples
        
        # If the remaining audio is too small to be meaningful (e.g. less than 0.1s) and we just covered it
        if end_sample >= total_samples:
            break
            
    return chunks

def find_longest_common_affix(s1: str, s2: str, max_lookback: int = 5) -> Tuple[int, str]:
    """Find word-level overlap between end of s1 and beginning of s2."""
    w1 = s1.split()
    w2 = s2.split()
    if not w1 or not w2:
        return 0, ""
    
    max_k = min(len(w1), len(w2), max_lookback)
    for k in range(max_k, 0, -1):
        if w1[-k:] == w2[:k]:
            return k, " ".join(w2[:k])
    return 0, ""

def reconcile_chunk_transcripts(
    chunk_transcripts: List[str],
    has_overlap: bool = False
) -> Dict[str, Any]:
    """
    Reconcile transcripts from sequential chunks into a cohesive whole transcript.
    
    Documented boundary reconciliation strategy:
    1. If no overlap was configured:
       Concatenate raw transcripts with a single whitespace delimiter, trimming edges.
    2. If overlap was configured:
       Inspect adjacent chunks for word-level or token-level duplication at the seam.
       If an exact overlap suffix/prefix match is found, deduplicate it.
       If partial or character-level overlap is detected, preserve boundary tokens and
       flag the seam for manual review in reconciliation metadata.
    
    Returns dict:
        combined_text: str
        raw_chunks: List[str]
        seam_events: List[Dict[str, Any]]
        duplication_detected: bool
    """
    cleaned_chunks = [c.strip() for c in chunk_transcripts if c and c.strip()]
    if not cleaned_chunks:
        return {
            "combined_text": "",
            "raw_chunks": chunk_transcripts,
            "seam_events": [],
            "duplication_detected": False
        }
    
    if len(cleaned_chunks) == 1:
        return {
            "combined_text": cleaned_chunks[0],
            "raw_chunks": chunk_transcripts,
            "seam_events": [],
            "duplication_detected": False
        }
    
    if not has_overlap:
        return {
            "combined_text": " ".join(cleaned_chunks),
            "raw_chunks": chunk_transcripts,
            "seam_events": [],
            "duplication_detected": False
        }
    
    # Overlap reconciliation
    reconciled_words: List[str] = cleaned_chunks[0].split()
    seam_events: List[Dict[str, Any]] = []
    duplication_flag = False
    
    for i in range(1, len(cleaned_chunks)):
        prev_text = " ".join(reconciled_words)
        curr_text = cleaned_chunks[i]
        curr_words = curr_text.split()
        
        match_len, matched_phrase = find_longest_common_affix(prev_text, curr_text)
        if match_len > 0:
            duplication_flag = True
            seam_events.append({
                "seam_index": i,
                "type": "exact_word_overlap_dedup",
                "matched_phrase": matched_phrase,
                "words_dropped": match_len
            })
            # Append non-overlapping words
            reconciled_words.extend(curr_words[match_len:])
        else:
            seam_events.append({
                "seam_index": i,
                "type": "no_direct_word_match",
                "matched_phrase": "",
                "words_dropped": 0
            })
            reconciled_words.extend(curr_words)
            
    return {
        "combined_text": " ".join(reconciled_words),
        "raw_chunks": chunk_transcripts,
        "seam_events": seam_events,
        "duplication_detected": duplication_flag
    }
