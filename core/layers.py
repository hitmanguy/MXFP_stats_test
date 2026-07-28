"""
core/layers.py
==============
FakeQuant wrapper layers for PyTorch models.

FakeQuantLinear         – wraps nn.Linear
FakeQuantGPT2Conv1D     – wraps transformers Conv1D (weights stored transposed!)
FakeQuantConv1d         – wraps nn.Conv1d (Wav2Vec2 feature extractor)

All layers:
  - Keep the original weight/bias in place (model stays on CPU/GPU as loaded)
  - Apply fake-quant to weights once (or per forward for activations)
  - Accept quant_mode for weights and activations independently
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from core.quantizer import (
    MXFP4Quantizer,
    fake_quant_mxfp4,
    fake_quant_mxfp4_residual,
    fake_quant_mxfp4_adaptive,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_weight_quantizer(weight_mode: str) -> Optional[MXFP4Quantizer]:
    if weight_mode in ("fp32", "bf16", "none"):
        return None
    return MXFP4Quantizer(mode=weight_mode)


def _quantise_weight(
    weight: torch.Tensor,
    weight_mode: str,
    block_size: int = 32,
) -> Tuple[torch.Tensor, float]:
    """Apply weight quantisation based on mode. Returns (quantised_w, trigger_rate)."""
    if weight_mode in ("fp32", "bf16", "none"):
        return weight, 0.0
    # ── MXFP4 family
    if weight_mode == "mxfp4":
        return fake_quant_mxfp4(weight, block_size), 0.0
    if weight_mode in ("mxfp4_residual", "mxfp4_residual_full", "mxfp4_residual_full_w"):
        return fake_quant_mxfp4_residual(weight, block_size), 1.0
    if weight_mode.startswith("mxfp4_adaptive_"):
        thresh = float(weight_mode.split("_")[-1])
        return fake_quant_mxfp4_adaptive(weight, thresh, block_size)
    # ── MXFP8 family
    if weight_mode == "mxfp8_e4m3":
        from core.quantizer import fake_quant_mxfp8_e4m3
        return fake_quant_mxfp8_e4m3(weight, block_size), 0.0
    if weight_mode == "mxfp8_e5m2":
        from core.quantizer import fake_quant_mxfp8_e5m2
        return fake_quant_mxfp8_e5m2(weight, block_size), 0.0
    if weight_mode == "mxfp8_e4m3_residual":
        from core.quantizer import fake_quant_mxfp8_e4m3_residual
        return fake_quant_mxfp8_e4m3_residual(weight, block_size), 1.0
    if weight_mode == "mxfp8_e5m2_residual":
        from core.quantizer import fake_quant_mxfp8_e5m2_residual
        return fake_quant_mxfp8_e5m2_residual(weight, block_size), 1.0
    if weight_mode.startswith("mxfp8_e4m3_adaptive_"):
        from core.quantizer import fake_quant_mxfp8_e4m3_adaptive
        thresh = float(weight_mode.split("_")[-1])
        return fake_quant_mxfp8_e4m3_adaptive(weight, thresh, block_size)
    if weight_mode.startswith("mxfp8_e5m2_adaptive_"):
        from core.quantizer import fake_quant_mxfp8_e5m2_adaptive
        thresh = float(weight_mode.split("_")[-1])
        return fake_quant_mxfp8_e5m2_adaptive(weight, thresh, block_size)
    # ── NVFP4 (stub — will raise)
    if weight_mode == "nvfp4":
        from core.quantizer import fake_quant_nvfp4
        return fake_quant_nvfp4(weight, block_size=16), 0.0
    return weight, 0.0


def _quantise_activation(
    x: torch.Tensor,
    act_mode: str,
    block_size: int = 32,
) -> Tuple[torch.Tensor, float]:
    """Apply activation quantisation based on mode. Returns (quantised_x, trigger_rate)."""
    if act_mode in ("fp32", "bf16", "none"):
        return x, 0.0

    numel = x.numel()
    pad_len = (block_size - (numel % block_size)) % block_size
    orig_shape = x.shape
    if pad_len > 0:
        x = torch.nn.functional.pad(x.flatten(), (0, pad_len))

    # ── MXFP4 family
    if act_mode == "mxfp4":
        out, rate = fake_quant_mxfp4(x, block_size), 0.0
    elif act_mode in ("mxfp4_residual", "mxfp4_residual_full", "mxfp4_residual_full_a"):
        out, rate = fake_quant_mxfp4_residual(x, block_size), 1.0
    elif act_mode.startswith("mxfp4_adaptive_"):
        thresh = float(act_mode.split("_")[-1])
        out, rate = fake_quant_mxfp4_adaptive(x, sqnr_thresh_db=thresh, block_size=block_size)
    # ── MXFP8 family
    elif act_mode == "mxfp8_e4m3":
        from core.quantizer import fake_quant_mxfp8_e4m3
        out, rate = fake_quant_mxfp8_e4m3(x, block_size), 0.0
    elif act_mode == "mxfp8_e5m2":
        from core.quantizer import fake_quant_mxfp8_e5m2
        out, rate = fake_quant_mxfp8_e5m2(x, block_size), 0.0
    elif act_mode == "mxfp8_e4m3_residual":
        from core.quantizer import fake_quant_mxfp8_e4m3_residual
        out, rate = fake_quant_mxfp8_e4m3_residual(x, block_size), 1.0
    elif act_mode == "mxfp8_e5m2_residual":
        from core.quantizer import fake_quant_mxfp8_e5m2_residual
        out, rate = fake_quant_mxfp8_e5m2_residual(x, block_size), 1.0
    elif act_mode.startswith("mxfp8_e4m3_adaptive_"):
        from core.quantizer import fake_quant_mxfp8_e4m3_adaptive
        thresh = float(act_mode.split("_")[-1])
        out, rate = fake_quant_mxfp8_e4m3_adaptive(x, sqnr_thresh_db=thresh, block_size=block_size)
    elif act_mode.startswith("mxfp8_e5m2_adaptive_"):
        from core.quantizer import fake_quant_mxfp8_e5m2_adaptive
        thresh = float(act_mode.split("_")[-1])
        out, rate = fake_quant_mxfp8_e5m2_adaptive(x, sqnr_thresh_db=thresh, block_size=block_size)
    # ── NVFP4
    elif act_mode == "nvfp4":
        from core.quantizer import fake_quant_nvfp4
        out, rate = fake_quant_nvfp4(x, block_size=16), 0.0
    else:
        raise ValueError(f"Unknown activation quant mode: {act_mode}")

    if pad_len > 0:
        out = out[:numel].reshape(orig_shape)
        
    return out, rate


# ─────────────────────────────────────────────────────────────────────────────
# FakeQuantLinear
# ─────────────────────────────────────────────────────────────────────────────

class FakeQuantLinear(nn.Module):
    """
    Drop-in fake-quant replacement for nn.Linear.

    Weight layout: [out_features, in_features]  (standard nn.Linear convention)
    Blocking is applied along in_features dimension (rows of the weight matrix
    are already correctly oriented for 32-element blocks).
    """

    def __init__(
        self,
        original: nn.Linear,
        weight_mode: str = "mxfp4",
        act_mode: str = "mxfp4",
        block_size: int = 32,
    ):
        super().__init__()
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.weight = nn.Parameter(original.weight.data.clone())
        self.bias = nn.Parameter(original.bias.data.clone()) if original.bias is not None else None
        self.weight_mode = weight_mode
        self.act_mode = act_mode
        self.block_size = block_size

        # Trigger rate tracking (updated each forward pass)
        self.last_weight_trigger_rate: float = 0.0
        self.last_act_trigger_rate: float = 0.0

        # Pre-quantise weights once (weights don't change at runtime).
        if weight_mode not in ("fp32", "bf16", "none"):
            w, rate = _quantise_weight(self.weight.data, weight_mode, block_size)
            self.weight.data = w.to(self.weight.data.dtype)
            self.last_weight_trigger_rate = rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dev = x.device
        # ── Weight ──────────────────────────────────────────────────────────
        w = self.weight.to(device=dev, dtype=x.dtype)

        # ── Activation ──────────────────────────────────────────────────────
        x_q, act_rate = _quantise_activation(x.float(), self.act_mode, self.block_size)
        x_q = x_q.to(device=dev, dtype=x.dtype)
        self.last_act_trigger_rate = act_rate

        # ── Linear ──────────────────────────────────────────────────────────
        bias = self.bias.to(device=dev, dtype=x.dtype) if self.bias is not None else None
        return F.linear(x_q, w, bias)

    @classmethod
    def from_linear(
        cls,
        module: nn.Linear,
        weight_mode: str = "mxfp4",
        act_mode: str = "mxfp4",
        block_size: int = 32,
    ) -> "FakeQuantLinear":
        return cls(module, weight_mode, act_mode, block_size)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"w={self.weight_mode}, a={self.act_mode}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# FakeQuantGPT2Conv1D
# ─────────────────────────────────────────────────────────────────────────────

class FakeQuantGPT2Conv1D(nn.Module):
    """
    Drop-in fake-quant replacement for HuggingFace GPT-2's Conv1D layer.

    CRITICAL: HuggingFace GPT-2 Conv1D stores weight as [in_features, out_features]
    — i.e. TRANSPOSED relative to nn.Linear's [out_features, in_features].
    The forward pass is:  output = x @ weight + bias   (no explicit transpose)

    When forming 32-element blocks we MUST group along the in_features axis
    (axis-0 of the stored weight, which has shape [in_features, out_features]).
    We do this by:
      1. Transposing stored weight → [out_features, in_features]  (nn.Linear layout)
      2. Quantising rows of [out_features, in_features] (each row = one output neuron's
         full in_features input weights — contiguous in memory, correct locality)
      3. Transposing back → [in_features, out_features]  for the matmul
    """

    def __init__(
        self,
        original,  # transformers.pytorch_utils.Conv1D
        weight_mode: str = "mxfp4",
        act_mode: str = "mxfp4",
        block_size: int = 32,
    ):
        super().__init__()
        # original.weight shape: [in_features, out_features]
        self.weight = nn.Parameter(original.weight.data.clone())
        self.bias = nn.Parameter(original.bias.data.clone()) if original.bias is not None else None
        self.in_features = original.weight.shape[0]    # n_state_in
        self.out_features = original.weight.shape[1]   # n_state_out
        self.weight_mode = weight_mode
        self.act_mode = act_mode
        self.block_size = block_size

        self.last_weight_trigger_rate: float = 0.0
        self.last_act_trigger_rate: float = 0.0

        # Pre-quantise weight (transpose → quant → transpose back).
        if weight_mode not in ("fp32", "bf16", "none"):
            sq = self._quant_weight(self.weight.data, weight_mode)
            self.weight.data = sq.to(self.weight.data.dtype)

    def _quant_weight(self, w: torch.Tensor, mode: str) -> torch.Tensor:
        """
        w is stored as [in_features, out_features].
        Transpose to [out_features, in_features], quantise along in_features
        dimension (blocks of 32 contiguous input features), then transpose back.
        """
        w_t = w.t().contiguous()        # [out_features, in_features]
        w_q, rate = _quantise_weight(w_t, mode, self.block_size)
        self.last_weight_trigger_rate = rate
        return w_q.t().contiguous()     # [in_features, out_features]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dev = x.device
        # ── Weight ──────────────────────────────────────────────────────────
        w = self.weight.to(device=dev, dtype=x.dtype)

        # ── Activation ──────────────────────────────────────────────────────
        x_q, act_rate = _quantise_activation(x.float(), self.act_mode, self.block_size)
        x_q = x_q.to(device=dev, dtype=x.dtype)
        self.last_act_trigger_rate = act_rate

        # ── Matmul (Conv1D convention: output = x @ weight + bias) ──────────
        bias = self.bias.to(device=dev, dtype=x.dtype) if self.bias is not None else None
        size_out = x_q.shape[:-1] + (self.out_features,)
        out = torch.addmm(bias, x_q.view(-1, x_q.shape[-1]), w) if bias is not None \
              else x_q.view(-1, x_q.shape[-1]) @ w
        return out.view(size_out)

    @classmethod
    def from_conv1d(
        cls,
        module,
        weight_mode: str = "mxfp4",
        act_mode: str = "mxfp4",
        block_size: int = 32,
    ) -> "FakeQuantGPT2Conv1D":
        return cls(module, weight_mode, act_mode, block_size)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"w={self.weight_mode}, a={self.act_mode} [Conv1D layout]"
        )


# ─────────────────────────────────────────────────────────────────────────────
# FakeQuantConv1d  (nn.Conv1d — Wav2Vec2 feature extractor)
# ─────────────────────────────────────────────────────────────────────────────

class FakeQuantConv1d(nn.Module):
    """
    Drop-in fake-quant replacement for torch.nn.Conv1d.

    Used by Wav2Vec2's convolutional feature extractor, which operates on raw
    waveform samples.  Weight layout: [out_channels, in_channels/groups, kW].

    Blocking strategy: flatten to [out_channels, in_channels/groups * kW] and
    quantise rows (each output channel's full input kernel), matching the same
    per-output-row blocking used in FakeQuantLinear.
    """

    def __init__(
        self,
        original: nn.Conv1d,
        weight_mode: str = "mxfp4",
        act_mode: str = "mxfp4",
        block_size: int = 32,
    ):
        super().__init__()
        self.weight = nn.Parameter(original.weight.data.clone())
        self.bias = (
            nn.Parameter(original.bias.data.clone())
            if original.bias is not None
            else None
        )
        self.stride = original.stride
        self.padding = original.padding
        self.dilation = original.dilation
        self.groups = original.groups
        self.weight_mode = weight_mode
        self.act_mode = act_mode
        self.block_size = block_size

        self.last_weight_trigger_rate: float = 0.0
        self.last_act_trigger_rate: float = 0.0

        # Pre-quantise weights once.  Flatten kernel dimension for blocking.
        if weight_mode not in ("fp32", "bf16", "none"):
            original_shape = self.weight.data.shape   # [out, in/g, kW]
            w_flat = self.weight.data.flatten(1)      # [out, in/g * kW]
            w_q, rate = _quantise_weight(w_flat, weight_mode, block_size)
            self.weight.data = w_q.reshape(original_shape).to(self.weight.data.dtype)
            self.last_weight_trigger_rate = rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dev = x.device
        # ── Weight ──────────────────────────────────────────────────────────
        w = self.weight.to(device=dev, dtype=x.dtype)

        # ── Activation ──────────────────────────────────────────────────────
        # x is [batch, channels, length]. Transpose to [batch, length, channels]
        # so that quantization blocks along the channel dimension.
        x_t = x.transpose(1, 2).contiguous()
        x_q_t, act_rate = _quantise_activation(x_t.float(), self.act_mode, self.block_size)
        x_q = x_q_t.transpose(1, 2).contiguous()
        x_q = x_q.to(device=dev, dtype=x.dtype)
        self.last_act_trigger_rate = act_rate

        # ── Conv1d ──────────────────────────────────────────────────────────
        bias = self.bias.to(device=dev, dtype=x.dtype) if self.bias is not None else None
        return F.conv1d(x_q, w, bias, self.stride, self.padding, self.dilation, self.groups)

    @classmethod
    def from_conv1d(
        cls,
        module: nn.Conv1d,
        weight_mode: str = "mxfp4",
        act_mode: str = "mxfp4",
        block_size: int = 32,
    ) -> "FakeQuantConv1d":
        return cls(module, weight_mode, act_mode, block_size)

    def extra_repr(self) -> str:
        oc, ic_g, kw = self.weight.shape
        return (
            f"out={oc}, in/g={ic_g}, kW={kw}, "
            f"w={self.weight_mode}, a={self.act_mode} [Conv1d layout]"
        )
