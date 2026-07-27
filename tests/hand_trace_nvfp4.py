"""
tests/hand_trace_nvfp4.py
=========================
Hand-trace of TRUE NVFP4 (two-level scaling) on a real GPT-2 weight tensor.

We trace the FULL first weight matrix (transformer.h[0].attn.c_attn.weight, shape [768, 2304])
to get a realistic global_scale, then zero in on the first 16-element block (first row,
first 16 values) for the per-element walkthrough.

Shows:
  tensor_amax → global_scale (FP32)
  → for block [0]: block_amax, block_scale_raw, nearest E4M3 block_scale_e4m3
  → per-element: x / (global_scale * block_scale) → round_e2m1 → dequant → error

Then verifies against PyTorch implementation for all 16 elements.
Also cross-checks 3 blocks (first, middle, last non-zero) to ensure the global scale
is consistent and no block overflows E4M3 range.
"""
from __future__ import annotations
import os, sys, math
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except ImportError:
    pass

import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LEVELS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
TIES = [
    (0.25,  0, 1, 0),
    (0.75,  1, 2, 2),
    (1.25,  2, 3, 2),
    (1.75,  3, 4, 4),
    (2.5,   4, 5, 4),
    (3.5,   5, 6, 6),
    (5.0,   6, 7, 6),
]
MXFP4_FORMAT_MAX = 6.0
MXFP8_E4M3_FORMAT_MAX = 448.0


def ref_round_e2m1(x_abs: float) -> tuple:
    x = min(x_abs, MXFP4_FORMAT_MAX)
    for (mid, lo, hi, winner) in TIES:
        if x == mid:
            return LEVELS[winner], f"TIE@{mid}→{LEVELS[winner]}"
    best_dist, best = float("inf"), 0.0
    for lv in LEVELS:
        d = abs(x - lv)
        if d < best_dist:
            best_dist, best = d, lv
    return best, f"nearest={best}(dist={best_dist:.6f})"


def ref_quantize_block_nvfp4(block16: list, global_scale: float, verbose: bool = False) -> tuple:
    """Pure-Python NVFP4 block quantize. Returns (dequant_vals, block_scale_e4m3)."""
    block_amax = max(abs(v) for v in block16)
    block_scale_raw = block_amax / (global_scale * MXFP4_FORMAT_MAX) if block_amax > 0 else 0.0
    # Clamp and round to nearest E4M3
    block_scale_e4m3_t = torch.tensor([block_scale_raw], dtype=torch.float32).clamp(0, MXFP8_E4M3_FORMAT_MAX)
    block_scale_e4m3 = block_scale_e4m3_t.to(torch.float8_e4m3fn).to(torch.float32).item()
    safe_scale = block_scale_e4m3 if block_scale_e4m3 > 0 else 1e-38

    if verbose:
        print(f"  block_amax          = {block_amax:.8f}")
        print(f"  block_scale_raw     = block_amax / (global_scale * 6.0)")
        print(f"                      = {block_amax:.8f} / ({global_scale:.8e} * 6.0)")
        print(f"                      = {block_scale_raw:.8f}")
        print(f"  nearest E4M3 value  = {block_scale_e4m3:.8f}")
        print(f"  total_scale         = global_scale * block_scale_e4m3")
        print(f"                      = {global_scale:.8e} * {block_scale_e4m3:.8f}")
        print(f"                      = {global_scale * safe_scale:.8e}")
        print()

    dequant = []
    for i, v in enumerate(block16):
        sign = 1.0 if v >= 0.0 else -1.0
        x_scaled = abs(v) / (global_scale * safe_scale)
        level, note = ref_round_e2m1(x_scaled)
        dq = level * sign * global_scale * safe_scale
        if verbose:
            print(f"  [{i:2d}] raw={v:+.6f}  /total_scale→{x_scaled:9.5f}  E2M1={level:.1f}  dq={dq:+.6f}  ({note})")
        dequant.append(dq)
    return dequant, block_scale_e4m3


def main():
    from core.quantizer import fake_quant_nvfp4, MXFP8_E4M3_FORMAT_MAX
    from transformers import AutoModelForCausalLM

    print("=" * 72)
    print("  NVFP4 HAND TRACE — transformer.h[0].attn.c_attn.weight")
    print("  Two-level: global FP32 scale + per-block E4M3 scale + E2M1 elements")
    print("=" * 72)

    model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32)
    model.eval()
    w_full = model.transformer.h[0].attn.c_attn.weight.detach().float()  # [768, 2304]
    w_flat = w_full.reshape(-1)
    N = w_flat.numel()
    print(f"\n  Weight shape: {list(w_full.shape)}  ({N:,} elements)")

    # ── Global scale ──────────────────────────────────────────────────────────
    tensor_amax = w_flat.abs().max().item()
    global_scale = tensor_amax / (MXFP4_FORMAT_MAX * MXFP8_E4M3_FORMAT_MAX)
    print(f"\n  tensor_amax         = {tensor_amax:.8f}")
    print(f"  global_scale        = tensor_amax / (6.0 * 448.0)")
    print(f"                      = {tensor_amax:.8f} / 2688.0")
    print(f"                      = {global_scale:.8e}  (FP32)")

    # ── Verify no block overflows E4M3 ────────────────────────────────────────
    w_blocked = w_flat.reshape(-1, 16)
    B = w_blocked.shape[0]
    block_amaxes = w_blocked.abs().amax(dim=-1)
    block_scale_raws = block_amaxes / (global_scale * MXFP4_FORMAT_MAX)
    max_block_scale_raw = block_scale_raws.max().item()
    print(f"\n  Total blocks        = {B:,}")
    print(f"  Max block_scale_raw = {max_block_scale_raw:.6f}  (must be ≤ 448.0)")
    if max_block_scale_raw <= MXFP8_E4M3_FORMAT_MAX + 1e-6:
        print(f"  ✓ All block scales fit in E4M3 range [0, 448]")
    else:
        print(f"  !! OVERFLOW: block scale {max_block_scale_raw} > 448 — global scale is wrong!")

    # ── Detailed trace: first block (16 elements) ─────────────────────────────
    block0 = w_flat[:16].tolist()
    print(f"\n{'─'*72}")
    print("  DETAILED TRACE: Block 0 (first 16 elements)")
    print(f"{'─'*72}")
    print(f"\n  Raw values:")
    print("  " + "  ".join(f"{v:+.6f}" for v in block0))

    print("\n─── PURE PYTHON REFERENCE ───────────────────────────────────────────")
    ref_dq, ref_block_scale = ref_quantize_block_nvfp4(block0, global_scale, verbose=True)

    # ── PyTorch implementation ────────────────────────────────────────────────
    print("─── PYTORCH IMPLEMENTATION ──────────────────────────────────────────")
    t_block = torch.tensor(block0, dtype=torch.float32).unsqueeze(0)  # [1, 16]
    # Pass the FULL weight to get correct global scale
    pt_dq_full = fake_quant_nvfp4(w_flat)
    pt_dq_block0 = pt_dq_full[:16].tolist()

    mismatches = 0
    max_diff = 0.0
    for i, (r, p, raw) in enumerate(zip(ref_dq, pt_dq_block0, block0)):
        diff = abs(r - p)
        if diff > max_diff:
            max_diff = diff
        if diff > 1e-6:
            print(f"  [{i:2d}] raw={raw:+.6f}  ref={r:+.8f}  pt={p:+.8f}  !! diff={diff:.2e} !!")
            mismatches += 1
    if mismatches == 0:
        print(f"  ✓ All 16 elements match exactly (max_diff={max_diff:.2e})")
    else:
        print(f"  !! {mismatches} mismatches (max_diff={max_diff:.2e})")

    # ── Cross-check 2 more blocks ─────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("  CROSS-CHECK: 3 additional blocks (indices 100, 500, 1000)")
    print(f"{'─'*72}")
    total_mismatch = 0
    for blk_idx in [100, 500, 1000]:
        blk = w_flat[blk_idx*16:(blk_idx+1)*16].tolist()
        ref_dq2, ref_bs2 = ref_quantize_block_nvfp4(blk, global_scale, verbose=False)
        pt_dq2 = pt_dq_full[blk_idx*16:(blk_idx+1)*16].tolist()
        mm = sum(1 for r, p in zip(ref_dq2, pt_dq2) if abs(r - p) > 1e-6)
        total_mismatch += mm
        status = "✓" if mm == 0 else f"!! {mm} mismatches !!"
        ba = max(abs(v) for v in blk)
        bsr = ba / (global_scale * MXFP4_FORMAT_MAX)
        bse4m3_t = torch.tensor([bsr]).clamp(0, 448.0).to(torch.float8_e4m3fn).to(torch.float32).item()
        print(f"  Block {blk_idx:4d}: block_amax={ba:.5f}  scale_raw={bsr:.3f}  scale_e4m3={bse4m3_t:.3f}  {status}")
    if total_mismatch == 0:
        print(f"\n  ✓ All cross-check blocks match between reference and PyTorch")

    # ── Error analysis on first 64 blocks ─────────────────────────────────────
    import numpy as np
    first_n = min(64 * 16, N)
    raw_arr = w_flat[:first_n].numpy()
    dq_arr  = pt_dq_full[:first_n].numpy()
    err     = raw_arr - dq_arr
    print(f"\n{'─'*72}")
    print(f"  Error analysis — first {first_n} elements ({first_n//16} blocks)")
    print(f"{'─'*72}")
    print(f"  Max absolute error:  {np.abs(err).max():.8f}")
    print(f"  MSE:                 {np.mean(err**2):.10f}")
    print(f"  Relative Frobenius:  {np.linalg.norm(err)/(np.linalg.norm(raw_arr)+1e-30):.8f}")

    # Compare MSE vs MXFP4 (32-element blocks, E8M0) on same region
    from core.quantizer import fake_quant_mxfp4
    t_region = w_flat[:first_n]
    mse_nvfp4 = float(np.mean((w_flat[:first_n].numpy() - pt_dq_full[:first_n].numpy())**2))
    mse_mxfp4 = float(np.mean((w_flat[:first_n].numpy() - fake_quant_mxfp4(t_region).numpy())**2))
    print(f"\n  MSE comparison (first {first_n//16} blocks):")
    print(f"    MXFP4 (E2M1, E8M0, 32-elem blocks): {mse_mxfp4:.8f}")
    print(f"    NVFP4 (E2M1, E4M3, 16-elem blocks): {mse_nvfp4:.8f}")
    if mse_nvfp4 < mse_mxfp4:
        print(f"  ✓ NVFP4 MSE < MXFP4 MSE (finer E4M3 scale + half block size)")
    else:
        ratio = mse_nvfp4 / mse_mxfp4
        print(f"  NOTE: NVFP4 MSE {'>' if ratio > 1 else '<='} MXFP4 MSE (ratio={ratio:.2f})")
        print(f"        This may be expected if the weight has very smooth distribution")
        print(f"        (E8M0 power-of-two scales can be more efficient for near-zero weights)")

    return total_mismatch


if __name__ == "__main__":
    n = main()
    sys.exit(0 if n == 0 else 1)
