"""Unit tests for audio decoding, validation, and resampling."""

import unittest
import numpy as np
import io
import soundfile as sf
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from audio_utils import load_and_preprocess_audio, AudioProcessingError

class TestAudioUtils(unittest.TestCase):
    def test_load_and_resample(self):
        # 2 seconds of 44.1kHz audio
        sr = 44100
        t = np.linspace(0, 2.0, sr * 2, endpoint=False)
        sin_wave = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

        buf = io.BytesIO()
        sf.write(buf, sin_wave, sr, format="WAV")
        buf.seek(0)

        audio, orig_sr, final_sr, duration, ops = load_and_preprocess_audio(buf, target_sample_rate=16000)
        self.assertEqual(orig_sr, 44100)
        self.assertEqual(final_sr, 16000)
        self.assertAlmostEqual(duration, 2.0, places=2)
        self.assertEqual(len(audio), 32000)

    def test_empty_audio_raises(self):
        buf = io.BytesIO()
        empty = np.array([], dtype=np.float32)
        sf.write(buf, empty, 16000, format="WAV")
        buf.seek(0)
        with self.assertRaises(AudioProcessingError):
            load_and_preprocess_audio(buf)

if __name__ == "__main__":
    unittest.main()
