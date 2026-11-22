"""Unit tests for chunking and boundary reconciliation."""

import unittest
import numpy as np
import sys
import os

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from chunking import create_audio_chunks, find_longest_common_affix, reconcile_chunk_transcripts

class TestChunking(unittest.TestCase):
    def setUp(self):
        self.sr = 16000
        # 12.3 seconds of synthetic audio
        self.duration_sec = 12.3
        self.audio = np.zeros(int(self.duration_sec * self.sr), dtype=np.float32)

    def test_no_overlap_chunking(self):
        chunks = create_audio_chunks(self.audio, sample_rate=self.sr, chunk_duration_sec=5.0, overlap_sec=0.0)
        # 12.3s audio with 5s window, no overlap -> [0-5s], [5-10s], [10-12.3s] (3 chunks)
        self.assertEqual(len(chunks), 3)
        self.assertAlmostEqual(chunks[0]["start_sec"], 0.0)
        self.assertAlmostEqual(chunks[0]["end_sec"], 5.0)
        self.assertAlmostEqual(chunks[1]["start_sec"], 5.0)
        self.assertAlmostEqual(chunks[1]["end_sec"], 10.0)
        self.assertAlmostEqual(chunks[2]["start_sec"], 10.0)
        self.assertAlmostEqual(chunks[2]["end_sec"], 12.3, places=2)
        # Verify final chunk is not padded to 5s
        self.assertAlmostEqual(chunks[2]["duration_sec"], 2.3, places=2)

    def test_overlap_chunking(self):
        chunks = create_audio_chunks(self.audio, sample_rate=self.sr, chunk_duration_sec=5.0, overlap_sec=0.5)
        # Stride = 4.5s: 0-5s, 4.5-9.5s, 9.0-12.3s -> 3 chunks
        self.assertEqual(len(chunks), 3)
        self.assertAlmostEqual(chunks[0]["start_sec"], 0.0)
        self.assertAlmostEqual(chunks[0]["end_sec"], 5.0)
        self.assertAlmostEqual(chunks[1]["start_sec"], 4.5)
        self.assertAlmostEqual(chunks[1]["end_sec"], 9.5)
        self.assertAlmostEqual(chunks[2]["start_sec"], 9.0)
        self.assertAlmostEqual(chunks[2]["end_sec"], 12.3, places=2)

    def test_longest_common_affix(self):
        s1 = "ఇండియన్ ఇన్స్టిట్యూట్ ఆఫ్ సైన్స్"
        s2 = "ఆఫ్ సైన్స్ కెమికల్ ఇంజనీరింగ్"
        k, overlap_phrase = find_longest_common_affix(s1, s2, max_lookback=4)
        self.assertEqual(k, 2)
        self.assertEqual(overlap_phrase, "ఆఫ్ సైన్స్")

    def test_reconcile_without_overlap(self):
        chunks = ["ఇండియన్ ఇన్స్టిట్యూట్", "ఆఫ్ సైన్స్", "కెమికల్ ఇంజనీరింగ్"]
        res = reconcile_chunk_transcripts(chunks, has_overlap=False)
        self.assertEqual(res["combined_text"], "ఇండియన్ ఇన్స్టిట్యూట్ ఆఫ్ సైన్స్ కెమికల్ ఇంజనీరింగ్")
        self.assertEqual(len(res["seam_events"]), 0)

    def test_reconcile_with_overlap_deduplication(self):
        chunks = [
            "ఇండియన్ ఇన్స్టిట్యూట్ ఆఫ్ సైన్స్",
            "ఆఫ్ సైన్స్ కెమికల్ ఇంజనీరింగ్"
        ]
        res = reconcile_chunk_transcripts(chunks, has_overlap=True)
        self.assertEqual(res["combined_text"], "ఇండియన్ ఇన్స్టిట్యూట్ ఆఫ్ సైన్స్ కెమికల్ ఇంజనీరింగ్")
        self.assertEqual(len(res["seam_events"]), 1)
        self.assertTrue(res["duplication_detected"])

if __name__ == "__main__":
    unittest.main()
