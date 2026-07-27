"""
tests/hand_trace_mxfp8.py
==========================
Hand-traces ONE real GPT-2 weight block through both MXFP8 E4M3 and E5M2,
against independent pure-Python reference implementations.

Block: transformer.h[0].attn.c_attn.weight[0, :32]
(Same block as MXFP4 trace for direct comparison.)

Shows: raw values → block amax → FLOOR E8M0 scale → x/scale → nearest E4M3/E5M2
       level (from pure-Python bit-field arithmetic) → dequant → error vs raw
       then compares with PyTorch implementation output.
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


# ─────────────────────────────────────────────────────────────────────────────
# Pure Python reference implementations
# ─────────────────────────────────────────────────────────────────────────────

def _build_e4m3_levels_python():
    """All non-negative E4M3 magnitudes, sorted. Max must be 448.0."""
    vals = set()
    vals.add(0.0)
    # Subnormal: exp=0, mant=1..7 → 2^(1-7) * mant/8 = 2^(-6) * mant/8
    for mant in range(1, 8):
        vals.add((2 ** -6) * mant / 8)
    # Normal: exp=1..15, mant=0..7, EXCEPT (exp=15, mant=7) = NaN
    for exp in range(1, 16):
        for mant in range(0, 8):
            if exp == 15 and mant == 7:
                continue
            vals.add(2 ** (exp - 7) * (1 + mant / 8))
    return sorted(vals)


def _build_e5m2_levels_python():
    """All non-negative E5M2 magnitudes, sorted. Max must be 57344.0."""
    vals = set()
    vals.add(0.0)
    # Subnormal: exp=0, mant=1..3 → 2^(1-15) * mant/4 = 2^(-14) * mant/4
    for mant in range(1, 4):
        vals.add((2 ** -14) * mant / 4)
    # Normal: exp=1..30, mant=0..3
    for exp in range(1, 31):
        for mant in range(0, 4):
            vals.add(2 ** (exp - 15) * (1 + mant / 4))
    # exp=31 = Inf/NaN — excluded
    return sorted(vals)


def ref_nearest_level(x_abs: float, levels: list) -> tuple[float, str]:
    """Find nearest level using simple loop (no tie-break needed for 8-bit — ties essentially impossible)."""
    best_dist = float("inf")
    best_lv = 0.0
    for lv in levels:
        d = abs(x_abs - lv)
        if d < best_dist:
            best_dist = d
            best_lv = lv
    return best_lv, f"nearest={best_lv} (dist={best_dist:.8f})"


def ref_quantize_block_mxfp8(block: list, format_max: float, levels: list, verbose: bool = False) -> tuple:
    """Pure-Python MXFP8 block quantization. Returns (dequant_values, scale)."""
    amax = max(abs(v) for v in block)
    if amax == 0.0:
        return [0.0] * len(block), 1.0

    log2_format_max = math.floor(math.log2(format_max))
    scale_exp = math.floor(math.log2(amax)) - log2_format_max
    scale = 2.0 ** scale_exp

    if verbose:
        print(f"  amax              = {amax:.8f}")
        print(f"  floor(log2(amax)) = {math.floor(math.log2(amax))}")
        print(f"  floor(log2({format_max})) = {log2_format_max}")
        print(f"  scale_exp         = {scale_exp}")
        print(f"  scale             = {scale:.8f}")
        print()

    dequant = []
    for i, v in enumerate(block):
        sign = 1.0 if v >= 0.0 else -1.0
        x_scaled = abs(v) / scale
        x_clamped = min(x_scaled, format_max)
        level, note = ref_nearest_level(x_clamped, levels)
        dq = level * sign * scale
        if verbose:
            print(f"  [{i:2d}] raw={v:+10.6f}  /scale → {x_scaled:12.6f}  quant={level:.6f}  dq={dq:+10.6f}  ({note})")
        dequant.append(dq)

    return dequant, scale


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def get_real_gpt2_block():
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32)
    model.eval()
    w = model.transformer.h[0].attn.c_attn.weight
    return w[0, :32].detach().float().tolist()


def main():
    from core.quantizer import (
        fake_quant_mxfp8_e4m3, fake_quant_mxfp8_e5m2,
        MXFP8_E4M3_FORMAT_MAX, MXFP8_E5M2_FORMAT_MAX,
        _compute_e8m0_scale,
    )

    print("=" * 72)
    print("  MXFP8 HAND TRACE — transformer.h[0].attn.c_attn.weight[0, :32]")
    print("  (Same block as MXFP4 trace: amax=0.6891, scale=0.125 for MXFP4)")
    print("=" * 72)

    block = get_real_gpt2_block()
    e4m3_levels = _build_e4m3_levels_python()
    e5m2_levels = _build_e5m2_levels_python()

    # Sanity: confirm our pure-Python tables hit the documented maxima
    assert e4m3_levels[-1] == 448.0, f"E4M3 python table max={e4m3_levels[-1]}, expected 448.0"
    assert e5m2_levels[-1] == 57344.0, f"E5M2 python table max={e5m2_levels[-1]}, expected 57344.0"
    print(f"\n  ✓ Pure-Python E4M3 table: {len(e4m3_levels)} levels, max={e4m3_levels[-1]}")
    print(f"  ✓ Pure-Python E5M2 table: {len(e5m2_levels)} levels, max={e5m2_levels[-1]}")

    print("\n  Raw values:")
    for i in range(0, 32, 8):
        print("  " + "  ".join(f"{v:+.6f}" for v in block[i:i+8]))

    for fmt, format_max, levels, pt_fn in [
        ("E4M3", MXFP8_E4M3_FORMAT_MAX, e4m3_levels, fake_quant_mxfp8_e4m3),
        ("E5M2", MXFP8_E5M2_FORMAT_MAX, e5m2_levels, fake_quant_mxfp8_e5m2),
    ]:
        print(f"\n{'─'*72}")
        print(f"  FORMAT: MXFP8 {fmt} (format_max={format_max})")
        print(f"{'─'*72}")
        print(f"\n─── PURE PYTHON REFERENCE ({fmt}) ─────────────────────────────────────")
        ref_dq, ref_scale = ref_quantize_block_mxfp8(block, format_max, levels, verbose=True)

        print(f"\n─── PYTORCH IMPLEMENTATION ({fmt}) ──────────────────────────────────────")
        t_block = torch.tensor(block, dtype=torch.float32).unsqueeze(0)
        amax_t = t_block.abs().amax(dim=-1, keepdim=True)
        pt_scale = _compute_e8m0_scale(amax_t, format_max=format_max).item()
        pt_dq = pt_fn(t_block, block_size=32).squeeze(0).tolist()

        print(f"  scale (reference) = {ref_scale:.8f}")
        print(f"  scale (pytorch)   = {pt_scale:.8f}")
        if abs(pt_scale - ref_scale) > 1e-9:
            print(f"  !! SCALE MISMATCH !!")
        else:
            print(f"  ✓ Scales match exactly")

        mismatches = 0
        max_diff = 0.0
        for i, (r, p, raw) in enumerate(zip(ref_dq, pt_dq, block)):
            diff = abs(r - p)
            if diff > max_diff:
                max_diff = diff
            if diff > 1e-5:
                print(f"  [{i:2d}] raw={raw:+.6f}  ref={r:+.8f}  pt={p:+.8f}  !! diff={diff:.2e} !!")
                mismatches += 1

        if mismatches == 0:
            print(f"  ✓ ALL 32 values match between reference and PyTorch (max_diff={max_diff:.2e})")
        else:
            print(f"  !! {mismatches} mismatches (max_diff={max_diff:.2e})")

        import numpy as np
        raw_arr = np.array(block)
        ref_arr = np.array(ref_dq)
        err = raw_arr - ref_arr
        print(f"\n  Error analysis ({fmt}):")
        print(f"    Max absolute error:  {np.abs(err).max():.8f}")
        print(f"    MSE:                 {np.mean(err**2):.10f}")
        print(f"    Relative Frobenius:  {np.linalg.norm(err)/(np.linalg.norm(raw_arr)+1e-30):.8f}")

    # Comparison with MXFP4 (from earlier trace, same block)
    print(f"\n{'─'*72}")
    print("  COMPARISON: Same block, different formats")
    print(f"{'─'*72}")
    from core.quantizer import fake_quant_mxfp4
    import numpy as np
    raw_arr = np.array(block)
    t_block = torch.tensor(block, dtype=torch.float32).unsqueeze(0)

    mse_fp4  = float(np.mean((raw_arr - fake_quant_mxfp4(t_block).squeeze(0).numpy())**2))
    mse_e4m3 = float(np.mean((raw_arr - fake_quant_mxfp8_e4m3(t_block).squeeze(0).numpy())**2))
    mse_e5m2 = float(np.mean((raw_arr - fake_quant_mxfp8_e5m2(t_block).squeeze(0).numpy())**2))

    print(f"  MXFP4 E2M1  MSE: {mse_fp4:.8f}   (4 bits/element, format_max=6)")
    print(f"  MXFP8 E4M3  MSE: {mse_e4m3:.8f}   (8 bits/element, format_max=448)")
    print(f"  MXFP8 E5M2  MSE: {mse_e5m2:.8f}   (8 bits/element, format_max=57344)")
    if mse_e4m3 < mse_fp4:
        print("  ✓ MXFP8 E4M3 < MXFP4 in MSE (more bits = less noise)")
    else:
        print("  !! WARNING: MXFP8 E4M3 >= MXFP4 — unexpected for this block")
    if mse_e5m2 < mse_fp4:
        print("  ✓ MXFP8 E5M2 < MXFP4 in MSE (more bits = less noise)")


if __name__ == "__main__":
    main()
