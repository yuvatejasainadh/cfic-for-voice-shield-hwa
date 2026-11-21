"""Original PellaV2 Audio Deepfake Detector PyTorch Architecture & Inference Pipeline.

Model Source: https://huggingface.co/Sadanie/pellav2-audio-deepfake-detector
Architecture:
- Backbone: 300M Parameter Speech Backbone (Wav2Vec2BertModel / w2v-bert-2.0, 24 Conformer layers)
- Multi-layer Pooling: Learned Softmax Layer Weighting across all hidden states
- Head: Linear projection (1024 -> 1)
- Preprocessing: 20Hz highpass Butterworth, LUFS normalization (-23 LUFS), Kaldi povey 160-dim fbank
- Decision: Sigmoid probability >= 0.50 -> FAKE, < 0.50 -> GENUINE
"""

import gc
import logging
import os
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from scipy.signal import butter, sosfiltfilt
import pyloudnorm as pyln
from transformers import Wav2Vec2BertConfig, Wav2Vec2BertModel
import torchaudio.compliance.kaldi as K

logger = logging.getLogger("pellav2_model")

SR = 16000
CROP = 4 * SR
TARGET_LUFS = -23.0


class PellaV2OriginalDetector(nn.Module):
    """Original PellaV2 Audio Deepfake Detector matching Sadanie/pellav2-audio-deepfake-detector."""

    def __init__(self, config: Wav2Vec2BertConfig):
        super().__init__()
        self.backbone = Wav2Vec2BertModel(config)
        self.layer_weights = nn.Parameter(torch.zeros(self.backbone.config.num_hidden_layers + 1))
        self.head = nn.Linear(self.backbone.config.hidden_size, 1)

    @staticmethod
    def features(wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract 160-dimensional povey log-mel filterbank features via Kaldi compliance."""
        feats, lens = [], []
        with torch.autocast(device_type=wav.device.type, enabled=False):
            for x in wav.float():
                nz = torch.nonzero(x).flatten()
                n_s = int(nz[-1]) + 1 if len(nz) else len(x)
                f = K.fbank(
                    x[None, :max(n_s, 400)] * 32768.0,
                    num_mel_bins=80,
                    sample_frequency=SR,
                    dither=0.0,
                    window_type="povey",
                )
                f = (f - f.mean(0, keepdim=True)) / torch.sqrt(f.var(0, keepdim=True) + 1e-7)
                n = f.shape[0] // 2 * 2
                feats.append(f[:n].reshape(n // 2, 160))
                lens.append(n // 2)

        F = max(lens)
        out = torch.zeros(len(feats), F, 160, device=wav.device)
        mask = torch.zeros(len(feats), F, dtype=torch.long, device=wav.device)
        for i, (f, n) in enumerate(zip(feats, lens)):
            out[i, :n] = f
            mask[i, :n] = 1
        return out, mask

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """Forward pass computing raw logit. Shape: (batch_size, 1) -> squeezed to (batch_size,)."""
        feats, mask = self.features(wav)
        hs = self.backbone(
            input_features=feats,
            attention_mask=mask,
            output_hidden_states=True,
        ).hidden_states
        w = torch.softmax(self.layer_weights, dim=0)
        h = torch.stack(hs).mul(w[:, None, None, None]).sum(0)
        m = mask[:, :, None].float()
        h = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(h).squeeze(-1)


# Alias for backward compatibility
PellaV2Model = PellaV2OriginalDetector


def loudness(x: np.ndarray) -> float:
    """Integrated loudness calculation using ITU-R BS.1770 / pyloudnorm."""
    try:
        if len(x) >= int(0.4 * SR):
            v = pyln.Meter(SR).integrated_loudness(x.astype(np.float64))
            if np.isfinite(v):
                return float(v)
    except Exception:
        pass
    return float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9) - 0.691)


def preprocess_audio(wav: np.ndarray, sr: int) -> torch.Tensor:
    """Preprocess raw audio strictly according to original PellaV2 specifications.
    
    1. Mono averaging
    2. Resample to 16 kHz
    3. Mean zero-centering
    4. 20 Hz highpass Butterworth filtering
    5. Duration crop or zero-padding to 4.0 seconds (64,000 samples)
    6. -23.0 LUFS loudness normalization
    """
    import torchaudio

    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        wav = torchaudio.functional.resample(torch.from_numpy(wav), sr, SR).numpy()
    
    wav = wav - wav.mean()
    if len(wav) > 3 * SR // 100:
        wav = sosfiltfilt(butter(2, 20.0, btype="highpass", fs=SR, output="sos"), wav)
    
    wav = wav.astype(np.float32)
    if len(wav) > CROP:
        off = (len(wav) - CROP) // 2
        wav = wav[off: off + CROP]
    
    if len(wav):
        gain = float(np.clip(TARGET_LUFS - loudness(wav), -40.0, 40.0))
        wav = np.clip(wav * 10 ** (gain / 20), -1.0, 1.0).astype(np.float32)
    
    if len(wav) < CROP:
        wav = np.concatenate([wav, np.zeros(CROP - len(wav), np.float32)])
    
    return torch.from_numpy(wav).unsqueeze(0)  # Shape: (1, 64000)


def load_original_pellav2_checkpoint(
    weights_path: str,
    config_path: str,
    device: str = "cpu",
) -> PellaV2OriginalDetector:
    """Load the original non-quantized PellaV2 detector from PyTorch weights."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Original PellaV2 checkpoint not found at: {weights_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Wav2Vec2Bert configuration not found at: {config_path}")

    logger.info(f"Loading Wav2Vec2BertConfig from {config_path}...")
    config = Wav2Vec2BertConfig.from_json_file(config_path)

    logger.info(f"Instantiating PellaV2OriginalDetector on {device}...")
    model = PellaV2OriginalDetector(config).to(device)

    logger.info(f"Loading model state dict from {weights_path}...")
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state, strict=True)
    del state
    gc.collect()

    model.eval()
    logger.info("Original PellaV2 detector successfully loaded in evaluation mode.")
    return model
