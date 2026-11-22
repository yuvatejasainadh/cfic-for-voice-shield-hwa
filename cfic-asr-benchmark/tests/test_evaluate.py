"""Unit tests for evaluation metrics and Telugu normalization."""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from evaluate import normalize_indic_text, compute_corpus_metrics

class TestEvaluate(unittest.TestCase):
    def test_normalize_telugu_text(self):
        raw = "  ఇండియన్,  ఇన్స్టిట్యూట్: ఆఫ్ సైన్స్!   పూర్తి చేశారు.  "
        expected = "ఇండియన్ ఇన్స్టిట్యూట్ ఆఫ్ సైన్స్ పూర్తి చేశారు"
        norm = normalize_indic_text(raw)
        self.assertEqual(norm, expected)

    def test_preserve_telugu_diacritics(self):
        # Virama (్), matras (ి, ు, etc.)
        raw = "కెమికల్ ఇంజనీరింగ్ లో పి హెచ్ డి"
        norm = normalize_indic_text(raw)
        self.assertEqual(norm, raw)

    def test_metrics_calculation(self):
        refs = ["ఇండియన్ ఇన్స్టిట్యూట్ ఆఫ్ సైన్స్"]
        preds = ["ఇండియన్ ఇన్స్టిట్యూట్ సైన్స్"] # 1 deletion
        durations = [5.0]
        latencies = [1.25]
        success_flags = [True]

        metrics = compute_corpus_metrics(
            references=refs,
            predictions=preds,
            durations=durations,
            latencies=latencies,
            success_flags=success_flags,
            model_init_time=1.5
        )

        self.assertEqual(metrics["total_samples"], 1)
        self.assertEqual(metrics["successful_samples"], 1)
        # Reference has 4 words, prediction has 3 words -> 1 error / 4 = 0.25 WER
        self.assertAlmostEqual(metrics["corpus_wer"], 0.25)
        self.assertAlmostEqual(metrics["corpus_rtf"], 0.25)
        self.assertEqual(metrics["model_init_seconds"], 1.5)

if __name__ == "__main__":
    unittest.main()
