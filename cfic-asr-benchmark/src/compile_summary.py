"""Compile unified summary metrics across all benchmark runs."""

import os
import sys
import pandas as pd
import numpy as np

# Ensure src is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import compute_corpus_metrics, normalize_indic_text

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "results")

INIT_TIMES = {
    "whisper_small": 2.72,
    "whisper_large-v3-turbo": 12.80,
    "indicconformer_te_large": 12.03
}

PEAK_RAM = {
    "whisper_small": 2100.0,
    "whisper_large-v3-turbo": 4890.0,
    "indicconformer_te_large": 1640.0
}

def analyze_df(df_subset, model_name, eval_name, init_time, peak_ram):
    refs = df_subset["reference"].tolist()
    preds = df_subset["prediction"].tolist()
    durs = df_subset["audio_duration_seconds"].tolist()
    lats = df_subset["inference_seconds"].tolist()
    succ = df_subset["success"].tolist()
    
    metrics = compute_corpus_metrics(
        references=refs,
        predictions=preds,
        durations=durs,
        latencies=lats,
        success_flags=succ,
        model_init_time=init_time,
        peak_ram_mb=peak_ram
    )
    
    metrics["model_name"] = model_name
    metrics["eval_set"] = eval_name
    return metrics

def main():
    # Load files
    whisper_small_df = pd.read_csv(os.path.join(RESULTS_DIR, "whisper_telugu.csv"))
    whisper_turbo_df = pd.read_csv(os.path.join(RESULTS_DIR, "whisper_turbo_telugu.csv"))
    indic_20_df = pd.read_csv(os.path.join(RESULTS_DIR, "indicconformer_telugu.csv"))
    indic_100_df = pd.read_csv(os.path.join(RESULTS_DIR, "indicconformer_100_telugu.csv"))
    chunk_df = pd.read_csv(os.path.join(RESULTS_DIR, "chunk_results.csv"))

    summary_rows = []

    # 1. 100-sample IndicConformer Full Evaluation
    m100 = analyze_df(
        indic_100_df,
        "indicconformer_te_large",
        "IndicConformer Full (100 samples, Whole Audio)",
        INIT_TIMES["indicconformer_te_large"],
        PEAK_RAM["indicconformer_te_large"]
    )
    summary_rows.append(m100)

    # 2. 20-sample IndicConformer Evaluation
    # Whole audio
    i20_whole = indic_20_df[~indic_20_df["sample_id"].str.contains("_5s_")]
    m20_whole = analyze_df(
        i20_whole,
        "indicconformer_te_large",
        "IndicConformer (20 samples, Whole Audio)",
        INIT_TIMES["indicconformer_te_large"],
        PEAK_RAM["indicconformer_te_large"]
    )
    summary_rows.append(m20_whole)

    # 20-sample 5s no overlap
    i20_5s_no = indic_20_df[indic_20_df["sample_id"].str.contains("_5s_no_overlap")]
    m20_5s_no = analyze_df(
        i20_5s_no,
        "indicconformer_te_large_5s_no_overlap",
        "IndicConformer (20 samples, 5s No-Overlap Chunks)",
        INIT_TIMES["indicconformer_te_large"],
        PEAK_RAM["indicconformer_te_large"]
    )
    summary_rows.append(m20_5s_no)

    # 20-sample 5s with overlap
    i20_5s_ov = indic_20_df[indic_20_df["sample_id"].str.contains("_5s_with_overlap")]
    m20_5s_ov = analyze_df(
        i20_5s_ov,
        "indicconformer_te_large_5s_with_overlap",
        "IndicConformer (20 samples, 5s Overlap Reconciled)",
        INIT_TIMES["indicconformer_te_large"],
        PEAK_RAM["indicconformer_te_large"]
    )
    summary_rows.append(m20_5s_ov)

    # 3. Matched 5-Sample Comparison (telugu_00000 to telugu_00004)
    # Whole audio
    ws_5_whole = whisper_small_df[~whisper_small_df["sample_id"].str.contains("_5s_")]
    wt_5_whole = whisper_turbo_df[~whisper_turbo_df["sample_id"].str.contains("_5s_")]
    ic_5_whole = indic_20_df[~indic_20_df["sample_id"].str.contains("_5s_")].head(5)

    m_ws_whole = analyze_df(ws_5_whole, "whisper_small", "Matched 5-Sample (Whole Audio)", INIT_TIMES["whisper_small"], PEAK_RAM["whisper_small"])
    m_wt_whole = analyze_df(wt_5_whole, "whisper_large-v3-turbo", "Matched 5-Sample (Whole Audio)", INIT_TIMES["whisper_large-v3-turbo"], PEAK_RAM["whisper_large-v3-turbo"])
    m_ic_whole = analyze_df(ic_5_whole, "indicconformer_te_large", "Matched 5-Sample (Whole Audio)", INIT_TIMES["indicconformer_te_large"], PEAK_RAM["indicconformer_te_large"])

    summary_rows.extend([m_ic_whole, m_ws_whole, m_wt_whole])

    # Matched 5-sample: 5s no overlap
    ws_5_no = whisper_small_df[whisper_small_df["sample_id"].str.contains("_5s_no_overlap")]
    wt_5_no = whisper_turbo_df[whisper_turbo_df["sample_id"].str.contains("_5s_no_overlap")]
    ic_5_no = indic_20_df[indic_20_df["sample_id"].str.contains("_5s_no_overlap")].head(5)

    m_ws_no = analyze_df(ws_5_no, "whisper_small", "Matched 5-Sample (5s No Overlap)", INIT_TIMES["whisper_small"], PEAK_RAM["whisper_small"])
    m_wt_no = analyze_df(wt_5_no, "whisper_large-v3-turbo", "Matched 5-Sample (5s No Overlap)", INIT_TIMES["whisper_large-v3-turbo"], PEAK_RAM["whisper_large-v3-turbo"])
    m_ic_no = analyze_df(ic_5_no, "indicconformer_te_large", "Matched 5-Sample (5s No Overlap)", INIT_TIMES["indicconformer_te_large"], PEAK_RAM["indicconformer_te_large"])

    summary_rows.extend([m_ic_no, m_ws_no, m_wt_no])

    # Matched 5-sample: 5s with overlap
    ws_5_ov = whisper_small_df[whisper_small_df["sample_id"].str.contains("_5s_with_overlap")]
    wt_5_ov = whisper_turbo_df[whisper_turbo_df["sample_id"].str.contains("_5s_with_overlap")]
    ic_5_ov = indic_20_df[indic_20_df["sample_id"].str.contains("_5s_with_overlap")].head(5)

    m_ws_ov = analyze_df(ws_5_ov, "whisper_small", "Matched 5-Sample (5s With Overlap)", INIT_TIMES["whisper_small"], PEAK_RAM["whisper_small"])
    m_wt_ov = analyze_df(wt_5_ov, "whisper_large-v3-turbo", "Matched 5-Sample (5s With Overlap)", INIT_TIMES["whisper_large-v3-turbo"], PEAK_RAM["whisper_large-v3-turbo"])
    m_ic_ov = analyze_df(ic_5_ov, "indicconformer_te_large", "Matched 5-Sample (5s With Overlap)", INIT_TIMES["indicconformer_te_large"], PEAK_RAM["indicconformer_te_large"])

    summary_rows.extend([m_ic_ov, m_ws_ov, m_wt_ov])

    # Create Summary DataFrame
    cols_to_keep = [
        "eval_set", "model_name", "total_samples", "success_rate_percent",
        "corpus_wer", "corpus_cer", "mean_clip_wer", "mean_clip_cer",
        "mean_latency_seconds", "median_latency_seconds", "p95_latency_seconds",
        "corpus_rtf", "total_audio_duration_seconds", "total_inference_seconds",
        "model_init_seconds", "peak_ram_mb"
    ]

    clean_rows = []
    for r in summary_rows:
        row_dict = {col: r.get(col, None) for col in cols_to_keep}
        clean_rows.append(row_dict)

    summary_df = pd.DataFrame(clean_rows)
    summary_path = os.path.join(RESULTS_DIR, "benchmark_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path}")

    # Print clean formatted tables
    print("\n=== BENCHMARK SUMMARY TABLE ===")
    print(summary_df.to_string())

    # Chunk level analysis from chunk_results.csv
    print("\n=== CHUNK LEVEL GRANULAR ANALYSIS ===")
    for model in chunk_df["model_name"].unique():
        for strat in chunk_df["chunk_strategy"].unique():
            cdf = chunk_df[(chunk_df["model_name"] == model) & (chunk_df["chunk_strategy"] == strat)]
            if len(cdf) > 0:
                print(f"\nModel: {model} | Strategy: {strat} | Total Chunks: {len(cdf)}")
                print(f"  Mean Chunk Duration: {cdf['chunk_duration_sec'].mean():.2f}s")
                print(f"  Mean Chunk Latency:  {cdf['inference_seconds'].mean():.2f}s")
                print(f"  Mean Chunk RTF:      {cdf['real_time_factor'].mean():.4f}")
                # Analyze tail chunk (chunk_index == 1) vs first chunk (chunk_index == 0)
                c0 = cdf[cdf["chunk_index"] == 0]
                c1 = cdf[cdf["chunk_index"] == 1]
                if len(c0) > 0:
                    print(f"  Chunk 0 (5.0s) Latency: {c0['inference_seconds'].mean():.2f}s (RTF: {c0['real_time_factor'].mean():.4f})")
                if len(c1) > 0:
                    print(f"  Chunk 1 (Tail) Latency: {c1['inference_seconds'].mean():.2f}s (RTF: {c1['real_time_factor'].mean():.4f})")

if __name__ == "__main__":
    main()
