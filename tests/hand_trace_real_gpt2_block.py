"""
tests/hand_trace_real_gpt2_block.py
=====================================
Hand-traces ONE real GPT-2 weight block (first 32 values of transformer.h.0.attn.c_attn.weight)
against an independent, loop-based pure-Python reference.

Shows: raw values → amax → floor(log2(amax)) - floor(log2(6)) = scale_exp → scale
      → x/scale → per-element nearest MXFP4 level (with RNE tie-break shown) → dequant → error

Also verifies the same block through the PyTorch path and checks they match exactly.
"""
from __future__ import annotations
import os, sys
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except ImportError:
    pass

import math
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ──────────────────────────────────────────────────────────────────────────────
# Pure Python reference — no torch, no vectorisation, no lookup
# ──────────────────────────────────────────────────────────────────────────────
LEVELS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
# tie-break table: midpoints and which index wins (even-index rule)
#   midpoint  between  lower_idx  upper_idx  even_idx_wins
TIES = [
    (0.25,  0, 1, 0),   # even index=0  → choose levels[0]=0.0
    (0.75,  1, 2, 2),   # even index=2  → choose levels[2]=1.0
    (1.25,  2, 3, 2),   # even index=2  → choose levels[2]=1.0
    (1.75,  3, 4, 4),   # even index=4  → choose levels[4]=2.0
    (2.5,   4, 5, 4),   # even index=4  → choose levels[4]=2.0
    (3.5,   5, 6, 6),   # even index=6  → choose levels[6]=4.0
    (5.0,   6, 7, 6),   # even index=6  → choose levels[6]=4.0
]
FORMAT_MAX = 6.0


def ref_round_to_mxfp4(x_abs_scaled: float, show: bool = False) -> tuple[float, str]:
    """Pure-python nearest-even rounding to MXFP4 levels. Returns (level, note)."""
    x = min(x_abs_scaled, FORMAT_MAX)  # clamp to FORMAT_MAX

    # Check exact tie first
    for (mid, lo, hi, winner) in TIES:
        if x == mid:
            note = f"TIE at {mid}: levels[{lo}]={LEVELS[lo]} vs levels[{hi}]={LEVELS[hi]} → even_idx={winner} → {LEVELS[winner]}"
            return LEVELS[winner], note

    # Find nearest level
    best_dist = float("inf")
    best_idx = -1
    for i, lvl in enumerate(LEVELS):
        d = abs(x - lvl)
        if d < best_dist:
            best_dist = d
            best_idx = i

    note = f"nearest={LEVELS[best_idx]} (dist={best_dist:.6f})"
    return LEVELS[best_idx], note


def ref_compute_scale(amax: float) -> tuple[float, int]:
    """Pure-python FLOOR-based E8M0 scale.  Returns (scale, scale_exp)."""
    if amax == 0.0:
        return 1.0, 0
    log2_format_max = math.floor(math.log2(FORMAT_MAX))  # = 2
    scale_exp = math.floor(math.log2(amax)) - log2_format_max
    return 2.0 ** scale_exp, scale_exp


def ref_quantize_block(block: list[float], verbose: bool = False) -> tuple[list[float], float]:
    """Full pure-python block quantization.  Returns (dequant_values, scale)."""
    amax = max(abs(v) for v in block)
    scale, scale_exp = ref_compute_scale(amax)

    if verbose:
        print(f"  amax              = {amax:.8f}")
        print(f"  floor(log2(amax)) = {math.floor(math.log2(max(amax,1e-38)))}")
        print(f"  floor(log2(6.0))  = 2")
        print(f"  scale_exp         = {scale_exp}")
        print(f"  scale             = {scale:.8f}")
        print()

    dequant = []
    for i, v in enumerate(block):
        sign = 1.0 if v >= 0.0 else -1.0
        x_scaled = abs(v) / scale
        level, note = ref_round_to_mxfp4(x_scaled)
        dq = level * sign * scale
        if verbose:
            print(f"  [{i:2d}] raw={v:+10.6f}  /scale → {x_scaled:9.6f}  quant={level:.1f}  dq={dq:+10.6f}  ({note})")
        dequant.append(dq)

    return dequant, scale


# ──────────────────────────────────────────────────────────────────────────────
# Load real GPT-2 weights
# ──────────────────────────────────────────────────────────────────────────────
def get_real_gpt2_block() -> list[float]:
    """
    Return the first 32 values of the first Conv1D weight in GPT-2:
        transformer.h.0.attn.c_attn.weight
    Shape: (768, 2304). We take row 0, first 32 columns.
    This is a real, untouched FP32 weight block.
    """
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32)
    model.eval()

    # Conv1D stores weight transposed vs nn.Linear: shape [in, out]
    # .weight attribute: [768, 2304]  (in_features × out_features for Conv1D)
    # We pick the first 32 values of the flattened weight (row 0, cols 0..31)
    w = model.transformer.h[0].attn.c_attn.weight   # [768, 2304]
    block_raw = w[0, :32].detach().float().tolist()   # first 32 values of first row
    return block_raw, w


# ──────────────────────────────────────────────────────────────────────────────
# Main trace
# ──────────────────────────────────────────────────────────────────────────────
def main():
    from core.quantizer import fake_quant_mxfp4, _compute_e8m0_scale

    print("=" * 72)
    print("  HAND TRACE — Real GPT-2 Block: transformer.h[0].attn.c_attn.weight[0, :32]")
    print("=" * 72)

    block_raw, w = get_real_gpt2_block()

    print(f"\n  Raw values (all 32):")
    for i in range(0, 32, 8):
        row = "  " + "  ".join(f"{v:+.6f}" for v in block_raw[i:i+8])
        print(row)

    print("\n─── PURE PYTHON REFERENCE ───────────────────────────────────────────")
    ref_dq, ref_scale = ref_quantize_block(block_raw, verbose=True)

    print("\n─── PYTORCH IMPLEMENTATION ──────────────────────────────────────────")
    t_block = torch.tensor(block_raw, dtype=torch.float32).unsqueeze(0)  # [1, 32]
    amax_t = t_block.abs().amax(dim=-1, keepdim=True)
    pt_scale_t = _compute_e8m0_scale(amax_t)
    pt_scale = pt_scale_t.item()
    pt_dq = fake_quant_mxfp4(t_block, block_size=32).squeeze(0).tolist()

    print(f"  amax (pytorch)    = {amax_t.item():.8f}")
    print(f"  scale (pytorch)   = {pt_scale:.8f}")
    print(f"  scale (reference) = {ref_scale:.8f}")

    if abs(pt_scale - ref_scale) > 1e-9:
        print("  !! SCALE MISMATCH !!")
    else:
        print("  ✓ Scales match exactly")

    print("\n  Per-element comparison (python ref vs pytorch):")
    mismatches = 0
    for i, (r, p, raw) in enumerate(zip(ref_dq, pt_dq, block_raw)):
        ok = abs(r - p) < 1e-6
        if not ok:
            print(f"  [{i:2d}] raw={raw:+.6f}  ref={r:+.6f}  pt={p:+.6f}  !! MISMATCH !!")
            mismatches += 1

    if mismatches == 0:
        print("  ✓ ALL 32 values match exactly between reference and PyTorch.")
    else:
        print(f"  !! {mismatches} MISMATCHES found !!")

    print()
    print("─── ERROR ANALYSIS ──────────────────────────────────────────────────")
    import numpy as np
    raw_arr = np.array(block_raw)
    ref_arr = np.array(ref_dq)
    err_arr = raw_arr - ref_arr
    print(f"  Max absolute error:  {np.abs(err_arr).max():.6f}")
    print(f"  MSE:                 {np.mean(err_arr**2):.8f}")
    print(f"  Relative Frobenius:  {np.linalg.norm(err_arr) / (np.linalg.norm(raw_arr) + 1e-30):.6f}")

    return mismatches


if __name__ == "__main__":
    n = main()
    import sys; sys.exit(0 if n == 0 else 1)
