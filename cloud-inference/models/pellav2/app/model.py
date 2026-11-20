"""PellaV2 K12 INT8 PyTorch Architecture & Inference Pipeline.

Model Backbone: 12-Layer Conformer (1024-dim, 16 heads, 4096-dim FFN, depthwise kernel 31)
Quantization: INT8 Dynamic Quantization (LinearPackedParams)
Feature Input: 160-dim log-mel spectrogram
Calibration: Platt scaling sigmoid(a * logit + b)
"""

import json
import logging
import math
import os
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.nn.quantized.dynamic as nnqd

logger = logging.getLogger("pellav2_model")


class PointwiseWrapper(nn.Module):
    """Wrapper exposing .lin for dynamic quantized pointwise convolution."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.lin = nnqd.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


class ConformerConv(nn.Module):
    """Conformer Convolution Module with depthwise separable 1D convolution."""

    def __init__(self, d_model: int = 1024, kernel_size: int = 31):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = PointwiseWrapper(d_model, d_model * 2)
        self.glu = nn.GLU(dim=-1)
        self.depthwise_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            groups=d_model,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.depthwise_layer_norm = nn.LayerNorm(d_model)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = PointwiseWrapper(d_model, d_model)
        self.dropout = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.layer_norm(x)
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = x.transpose(1, 2)
        x = self.depthwise_conv(x)
        x = x.transpose(1, 2)
        x = self.depthwise_layer_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        return res + x


class ConformerFFN(nn.Module):
    """Conformer Feed-Forward Module (Macaron-style half-step FFN)."""

    def __init__(self, d_model: int = 1024, d_ff: int = 4096):
        super().__init__()
        self.intermediate_dense = nnqd.Linear(d_model, d_ff)
        self.intermediate_act_fn = nn.SiLU()
        self.intermediate_dropout = nn.Dropout(0.0)
        self.output_dense = nnqd.Linear(d_ff, d_model)
        self.output_dropout = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.intermediate_dense(x)
        x = self.intermediate_act_fn(x)
        x = self.intermediate_dropout(x)
        x = self.output_dense(x)
        x = self.output_dropout(x)
        return x


class ConformerSelfAttn(nn.Module):
    """Multi-Head Self-Attention with relative positional distance embedding."""

    def __init__(self, d_model: int = 1024, n_heads: int = 16, max_dist: int = 73):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.linear_q = nnqd.Linear(d_model, d_model)
        self.linear_k = nnqd.Linear(d_model, d_model)
        self.linear_v = nnqd.Linear(d_model, d_model)
        self.linear_out = nnqd.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.0)
        self.distance_embedding = nn.Embedding(max_dist, self.d_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.linear_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.linear_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.linear_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)

        # Relative positional distance bias
        pos_idx = torch.arange(T, device=x.device)
        rel_pos = pos_idx.unsqueeze(0) - pos_idx.unsqueeze(1)
        rel_pos = torch.clamp(rel_pos + 36, 0, 72)
        rel_emb = self.distance_embedding(rel_pos)
        pos_bias = torch.einsum("bhtd,tsd->bhts", q, rel_emb) / (self.d_head ** 0.5)
        scores = scores + pos_bias

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, C)
        return self.linear_out(out)


class ConformerLayer(nn.Module):
    """Complete Conformer Encoder Layer."""

    def __init__(self, d_model: int = 1024):
        super().__init__()
        self.ffn1_layer_norm = nn.LayerNorm(d_model)
        self.ffn1 = ConformerFFN(d_model, 4096)
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.self_attn_dropout = nn.Dropout(0.0)
        self.self_attn = ConformerSelfAttn(d_model, 16, 73)
        self.conv_module = ConformerConv(d_model, 31)
        self.ffn2_layer_norm = nn.LayerNorm(d_model)
        self.ffn2 = ConformerFFN(d_model, 4096)
        self.final_layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Macaron FFN1 (half-step)
        x = x + 0.5 * self.ffn1(self.ffn1_layer_norm(x))
        # Self-Attention
        x = x + self.self_attn_dropout(self.self_attn(self.self_attn_layer_norm(x)))
        # Convolution module
        x = self.conv_module(x)
        # Macaron FFN2 (half-step)
        x = x + 0.5 * self.ffn2(self.ffn2_layer_norm(x))
        # Final LayerNorm
        return self.final_layer_norm(x)


class FeatureProjection(nn.Module):
    """Projects 160-dim acoustic features to 1024-dim hidden representations."""

    def __init__(self, in_features: int = 160, d_model: int = 1024):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_features)
        self.projection = nnqd.Linear(in_features, d_model)
        self.dropout = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x = self.projection(x)
        return self.dropout(x)


class PellaBackbone(nn.Module):
    """12-Layer Conformer Backbone."""

    def __init__(self, in_features: int = 160, d_model: int = 1024, n_layers: int = 12):
        super().__init__()
        self.feature_projection = FeatureProjection(in_features, d_model)
        self.masked_spec_embed = nn.Parameter(torch.zeros(d_model))
        self.encoder = nn.Module()
        self.encoder.dropout = nn.Dropout(0.0)
        self.encoder.layers = nn.ModuleList([ConformerLayer(d_model) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.feature_projection(x)
        hidden_states = [x]
        for layer in self.encoder.layers:
            x = layer(x)
            hidden_states.append(x)
        return hidden_states


class PellaV2Model(nn.Module):
    """Complete PellaV2 K12 INT8 Deepfake Acoustic Detection Model."""

    def __init__(
        self,
        cal_a: float = 1.5635074958458532,
        cal_b: float = -0.08237249469081723,
    ):
        super().__init__()
        self.cal_a = cal_a
        self.cal_b = cal_b
        self.layer_weights = nn.Parameter(torch.zeros(13))
        self.backbone = PellaBackbone(160, 1024, 12)
        self.head = nnqd.Linear(1024, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Acoustic features of shape (B, T, 160)
            
        Returns:
            logit: Raw scalar logit tensor of shape (B, 1)
            calibrated_score: Platt-scaled probability in [0.0, 1.0] of shape (B, 1)
        """
        hidden_states = self.backbone(x)
        stacked = torch.stack(hidden_states, dim=0)  # (13, B, T, 1024)
        weights = F.softmax(self.layer_weights, dim=0).view(13, 1, 1, 1)
        pooled_layers = (stacked * weights).sum(dim=0)  # (B, T, 1024)
        temporal_pool = pooled_layers.mean(dim=1)  # (B, 1024)
        logit = self.head(temporal_pool)  # (B, 1)
        calibrated_score = torch.sigmoid(self.cal_a * logit + self.cal_b)
        return logit, calibrated_score


def extract_160_log_mel_features(
    audio: np.ndarray,
    sample_rate: int = 16000,
    target_sample_rate: int = 16000,
    n_mels: int = 160,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
) -> torch.Tensor:
    """Extract 160-dim log-mel filterbank features from raw audio array."""
    if audio is None or audio.size == 0:
        raise ValueError("Audio waveform is empty")

    # Resample to target rate if needed
    if sample_rate != target_sample_rate:
        from scipy.signal import resample_poly
        gcd = math.gcd(sample_rate, target_sample_rate)
        audio = resample_poly(audio, target_sample_rate // gcd, sample_rate // gcd).astype(np.float32)

    # Ensure 1D float32
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    audio = np.clip(audio.astype(np.float32), -1.0, 1.0)

    t_audio = torch.from_numpy(audio)
    if t_audio.ndim == 1:
        t_audio = t_audio.unsqueeze(0)

    # Compute STFT
    window = torch.hann_window(win_length)
    stft = torch.stft(
        t_audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
    )
    mag = torch.abs(stft)  # (1, 257, T)

    # Compute 160 Mel filterbank matrix
    n_freq = n_fft // 2 + 1
    mel_fb = torch.zeros(n_mels, n_freq)
    low_mel = 2595 * np.log10(1 + 0.0 / 700.0)
    high_mel = 2595 * np.log10(1 + (target_sample_rate / 2.0) / 700.0)
    mel_pts = np.linspace(low_mel, high_mel, n_mels + 2)
    freq_pts = 700.0 * (10.0 ** (mel_pts / 2595.0) - 1.0)
    bin_pts = np.floor((n_fft + 1) * freq_pts / target_sample_rate).astype(int)

    for m in range(1, n_mels + 1):
        f_l, f_c, f_r = bin_pts[m - 1], bin_pts[m], bin_pts[m + 1]
        if f_c > f_l:
            mel_fb[m - 1, f_l:f_c] = (torch.arange(f_l, f_c) - f_l) / (f_c - f_l)
        if f_r > f_c:
            mel_fb[m - 1, f_c:f_r] = (f_r - torch.arange(f_c, f_r)) / (f_r - f_c)

    mel_spec = torch.matmul(mel_fb, mag)  # (1, 160, T)
    log_mel = torch.log(torch.clamp(mel_spec, min=1e-5)).transpose(1, 2)  # (1, T, 160)
    return log_mel


def load_pellav2_checkpoint(
    checkpoint_path: str,
    calibration_path: str | None = None,
) -> PellaV2Model:
    """Load PellaV2 K12 INT8 checkpoint safely and strictly into PyTorch module."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    cal_a = 1.5635074958458532
    cal_b = -0.08237249469081723

    if calibration_path and os.path.exists(calibration_path):
        try:
            with open(calibration_path, "r", encoding="utf-8") as f:
                cal_data = json.load(f)
                cal_a = float(cal_data.get("a", cal_a))
                cal_b = float(cal_data.get("b", cal_b))
                logger.info(f"Loaded calibration parameters: a={cal_a}, b={cal_b}")
        except Exception as exc:
            logger.warning(f"Failed to read calibration JSON: {exc}; using default constants.")

    model = PellaV2Model(cal_a=cal_a, cal_b=cal_b)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise ValueError(f"State dict mismatch: missing={missing}, unexpected={unexpected}")

    model.eval()
    logger.info(f"PellaV2 K12 INT8 successfully loaded from {checkpoint_path} (strict match).")
    return model
