"""
tests/test_mxfp8.py
===================
Unit tests for MXFP8 (E4M3 and E5M2) implementation.

Critical assertions:
  - E4M3 FORMAT_MAX = 448.0 (NOT 240.0 which is E4M3FNUZ)
  - E5M2 FORMAT_MAX = 57344.0
  - Scale formula is floor-based E8M0 (same as MXFP4, shared code path)
  - Residual pass reduces MSE vs primary-only
  - Adaptive trigger_rate is monotonically non-decreasing with threshold
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch

from core.quantizer import (
    MXFP8_E4M3_FORMAT_MAX,
    MXFP8_E5M2_FORMAT_MAX,
    _compute_e8m0_scale,
    fake_quant_mxfp8_e4m3,
    fake_quant_mxfp8_e5m2,
    fake_quant_mxfp8_e4m3_residual,
    fake_quant_mxfp8_e5m2_residual,
    fake_quant_mxfp8_e4m3_adaptive,
    fake_quant_mxfp8_e5m2_adaptive,
)


def print_section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. FORMAT_MAX assertions — the most critical test in this file
# ─────────────────────────────────────────────────────────────────────────────

def test_e4m3_format_max_is_448_not_240():
    """
    E4M3 FORMAT_MAX MUST be 448.0.
    240.0 would mean E4M3FNUZ (a different variant with no NaN/Inf and different bias).
    This project uses OCP MXFP8 E4M3 (float8_e4m3fn), which has max=448.0.
    """
    assert MXFP8_E4M3_FORMAT_MAX == 448.0, (
        f"E4M3 FORMAT_MAX is {MXFP8_E4M3_FORMAT_MAX} — expected 448.0. "
        "If this is 240.0, you have the FNUZ variant, which is WRONG for OCP MXFP8."
    )
    # Also verify that a large value clamped and cast reaches 448.0
    x = torch.tensor([1e6], dtype=torch.float32)
    x_clamped = x.clamp(-MXFP8_E4M3_FORMAT_MAX, MXFP8_E4M3_FORMAT_MAX)
    val = x_clamped.to(torch.float8_e4m3fn).to(torch.float32).item()
    assert val == 448.0, f"Expected 448.0, got {val}"
    print(f"  ✓ E4M3 FORMAT_MAX = {MXFP8_E4M3_FORMAT_MAX} (confirmed 448.0, not 240.0)")


def test_e5m2_format_max_is_57344():
    """E5M2 FORMAT_MAX must be 57344.0."""
    assert MXFP8_E5M2_FORMAT_MAX == 57344.0, (
        f"E5M2 FORMAT_MAX is {MXFP8_E5M2_FORMAT_MAX} — expected 57344.0."
    )
    x = torch.tensor([1e9], dtype=torch.float32)
    x_clamped = x.clamp(-MXFP8_E5M2_FORMAT_MAX, MXFP8_E5M2_FORMAT_MAX)
    val = x_clamped.to(torch.float8_e5m2).to(torch.float32).item()
    assert val == 57344.0, f"Expected 57344.0, got {val}"
    print(f"  ✓ E5M2 FORMAT_MAX = {MXFP8_E5M2_FORMAT_MAX} (confirmed 57344.0)")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Scale formula: same FLOOR-based E8M0, different log2_format_max
# ─────────────────────────────────────────────────────────────────────────────

def test_e4m3_scale_formula():
    """
    For amax=500: floor(log2(500)) - floor(log2(448)) = 8 - 8 = 0 → scale = 1.0
    For amax=1000: floor(log2(1000)) - 8 = 9 - 8 = 1 → scale = 2.0
    For amax=0.5: floor(log2(0.5)) - 8 = -1 - 8 = -9 → scale = 2^-9 ≈ 0.001953
    """
    cases = [
        (500.0,   1.0),    # floor(log2(500))=8,  8-8=0, 2^0=1
        (1000.0,  2.0),    # floor(log2(1000))=9, 9-8=1, 2^1=2
        (0.5,     2**-9),  # floor(log2(0.5))=-1, -1-8=-9
        (448.0,   1.0),    # floor(log2(448))=8, 8-8=0
        (449.0,   2.0),    # floor(log2(449))=8, 8-8=0 → actually still 1.0
    ]
    # Recalculate expected for 449.0
    # log2(449) = 8.81… → floor = 8; 8 - 8 = 0 → scale = 1.0
    cases[-1] = (449.0, 1.0)

    for amax_val, expected_scale in cases:
        amax = torch.tensor([[amax_val]])
        scale = _compute_e8m0_scale(amax, format_max=MXFP8_E4M3_FORMAT_MAX).item()
        assert abs(scale - expected_scale) < 1e-9, (
            f"amax={amax_val}: expected scale={expected_scale}, got {scale}"
        )
    print("  ✓ E4M3 scale formula (FLOOR-based E8M0) correct for all test cases")


def test_e5m2_scale_formula():
    """
    floor(log2(57344)) = 15
    amax=57344: 15-15=0 → scale=1.0
    amax=114688: floor(log2(114688))=16, 16-15=1 → scale=2.0
    """
    cases = [
        (57344.0, 1.0),
        (114688.0, 2.0),
        (1.0, 2**-15),   # floor(log2(1))=0, 0-15=-15
    ]
    for amax_val, expected_scale in cases:
        amax = torch.tensor([[amax_val]])
        scale = _compute_e8m0_scale(amax, format_max=MXFP8_E5M2_FORMAT_MAX).item()
        assert abs(scale - expected_scale) < 1e-9, (
            f"amax={amax_val}: expected scale={expected_scale}, got {scale}"
        )
    print("  ✓ E5M2 scale formula (FLOOR-based E8M0) correct for all test cases")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Roundtrip identity: fake_quant(0) = 0, fake_quant at exact level = that level
# ─────────────────────────────────────────────────────────────────────────────

def test_e4m3_zero_passthrough():
    x = torch.zeros(32)
    out = fake_quant_mxfp8_e4m3(x.unsqueeze(0))
    assert torch.all(out == 0.0), "All-zero block should stay zero"
    print("  ✓ E4M3 zero passthrough")


def test_e5m2_zero_passthrough():
    x = torch.zeros(32)
    out = fake_quant_mxfp8_e5m2(x.unsqueeze(0))
    assert torch.all(out == 0.0), "All-zero block should stay zero"
    print("  ✓ E5M2 zero passthrough")


def test_e4m3_exact_level_identity():
    """Values that are exact E4M3 levels should roundtrip exactly (within float32 precision)."""
    # amax = 1.0, scale_exp = floor(log2(1)) - 8 = -8, scale = 2^-8 = 1/256
    # scaled values: 0, 1, 2, ... but after dividing by 1/256 they'd be 0, 256, 512...
    # Better: choose values that are exact E4M3 representable in the scaled domain
    # With amax=1.0, scale = 2^(0-8) = 1/256. Values/scale = 256*val.
    # E4M3 with exp=0111+bias=7=exponent=0: normal, value = 2^(7-7)*(1+mant/8) = 1+mant/8
    # So exact levels near 1.0: 1.0, 1.125, 1.25, ... 1.875
    # In scaled domain: 256*1.0=256 etc. But we need those to also be E4M3 representable.
    # Simpler: use a block where amax is exactly E4M3-representable and chosen so scale=1.
    # With scale=1 (amax just below 448), E4M3 levels in [-448, 448] roundtrip exactly.
    exact_vals = [0.0, 0.001953125, 0.00390625, 1.0, 2.0, 4.0, 8.0, 16.0,
                   32.0, 64.0, 128.0, 256.0, 384.0, 448.0]
    # Pad to 32 elements  
    block = exact_vals + [0.0] * (32 - len(exact_vals))
    x = torch.tensor(block, dtype=torch.float32).unsqueeze(0)
    out = fake_quant_mxfp8_e4m3(x)
    # Check the non-zero exact levels roundtrip
    for v, q in zip(block[:len(exact_vals)], out.squeeze(0)[:len(exact_vals)].tolist()):
        assert abs(v - q) < 1e-4, f"E4M3 exact level {v} did not roundtrip: got {q}"
    print("  ✓ E4M3 exact levels roundtrip within tolerance")


# ─────────────────────────────────────────────────────────────────────────────
# 4. MXFP8 has lower error than MXFP4 (sanity check: more bits = less noise)
# ─────────────────────────────────────────────────────────────────────────────

def test_mxfp8_has_lower_mse_than_mxfp4():
    """MXFP8 (8 bits/element) should have lower quantization error than MXFP4 (4 bits)."""
    from core.quantizer import fake_quant_mxfp4
    torch.manual_seed(42)
    x = torch.randn(1, 1024)  # 32 blocks of 32

    mse_fp4 = ((x - fake_quant_mxfp4(x)) ** 2).mean().item()
    mse_e4m3 = ((x - fake_quant_mxfp8_e4m3(x)) ** 2).mean().item()
    mse_e5m2 = ((x - fake_quant_mxfp8_e5m2(x)) ** 2).mean().item()

    assert mse_e4m3 < mse_fp4, f"E4M3 MSE {mse_e4m3:.6f} >= MXFP4 MSE {mse_fp4:.6f}"
    assert mse_e5m2 < mse_fp4, f"E5M2 MSE {mse_e5m2:.6f} >= MXFP4 MSE {mse_fp4:.6f}"
    print(f"  ✓ MSE: MXFP4={mse_fp4:.6f}  E4M3={mse_e4m3:.6f}  E5M2={mse_e5m2:.6f}  (MXFP8 < MXFP4 ✓)")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Residual reduces MSE
# ─────────────────────────────────────────────────────────────────────────────

def test_e4m3_residual_reduces_mse():
    torch.manual_seed(42)
    x = torch.randn(1, 1024)
    mse_primary = ((x - fake_quant_mxfp8_e4m3(x)) ** 2).mean().item()
    mse_residual = ((x - fake_quant_mxfp8_e4m3_residual(x)) ** 2).mean().item()
    assert mse_residual < mse_primary, (
        f"E4M3 residual MSE {mse_residual:.8f} >= primary MSE {mse_primary:.8f}"
    )
    reduction = (1 - mse_residual / mse_primary) * 100
    print(f"  ✓ E4M3 residual MSE reduction: {mse_primary:.6f} → {mse_residual:.6f} ({reduction:.1f}%)")


def test_e5m2_residual_reduces_mse():
    torch.manual_seed(42)
    x = torch.randn(1, 1024)
    mse_primary = ((x - fake_quant_mxfp8_e5m2(x)) ** 2).mean().item()
    mse_residual = ((x - fake_quant_mxfp8_e5m2_residual(x)) ** 2).mean().item()
    assert mse_residual < mse_primary, (
        f"E5M2 residual MSE {mse_residual:.8f} >= primary MSE {mse_primary:.8f}"
    )
    reduction = (1 - mse_residual / mse_primary) * 100
    print(f"  ✓ E5M2 residual MSE reduction: {mse_primary:.6f} → {mse_residual:.6f} ({reduction:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Adaptive trigger_rate is monotonically non-decreasing with threshold
# ─────────────────────────────────────────────────────────────────────────────

def test_e4m3_adaptive_monotone():
    torch.manual_seed(99)
    x = torch.randn(1, 1024)
    thresholds = [10.0, 20.0, 30.0, 40.0, 100.0]
    prev_rate = -1.0
    for thresh in thresholds:
        _, rate = fake_quant_mxfp8_e4m3_adaptive(x, sqnr_thresh_db=thresh)
        assert rate >= prev_rate - 1e-6, (
            f"E4M3 trigger_rate not monotone: thresh={thresh}, rate={rate} < prev={prev_rate}"
        )
        prev_rate = rate
    print(f"  ✓ E4M3 adaptive trigger_rate monotonically non-decreasing (final rate={prev_rate:.3f})")


def test_e5m2_adaptive_monotone():
    torch.manual_seed(99)
    x = torch.randn(1, 1024)
    thresholds = [10.0, 20.0, 30.0, 40.0, 100.0]
    prev_rate = -1.0
    for thresh in thresholds:
        _, rate = fake_quant_mxfp8_e5m2_adaptive(x, sqnr_thresh_db=thresh)
        assert rate >= prev_rate - 1e-6, (
            f"E5M2 trigger_rate not monotone: thresh={thresh}, rate={rate} < prev={prev_rate}"
        )
        prev_rate = rate
    print(f"  ✓ E5M2 adaptive trigger_rate monotonically non-decreasing (final rate={prev_rate:.3f})")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Scale is genuinely shared code path (not a separate reimplementation)
# ─────────────────────────────────────────────────────────────────────────────

def test_scale_code_path_is_shared():
    """
    Verify that _compute_e8m0_scale with different format_max values gives
    the correct scale for each format, proving it's truly parametric.
    """
    from core.quantizer import MXFP4_FORMAT_MAX
    amax = torch.tensor([[100.0]])
    # MXFP4: floor(log2(100)) - floor(log2(6)) = 6 - 2 = 4 → scale = 16
    scale4 = _compute_e8m0_scale(amax, format_max=MXFP4_FORMAT_MAX).item()
    assert scale4 == 16.0, f"MXFP4 scale wrong: {scale4}"

    # E4M3: floor(log2(100)) - floor(log2(448)) = 6 - 8 = -2 → scale = 0.25
    scale8e4m3 = _compute_e8m0_scale(amax, format_max=MXFP8_E4M3_FORMAT_MAX).item()
    assert scale8e4m3 == 0.25, f"E4M3 scale wrong: {scale8e4m3}"

    # E5M2: floor(log2(100)) - floor(log2(57344)) = 6 - 15 = -9 → scale = 2^-9
    scale8e5m2 = _compute_e8m0_scale(amax, format_max=MXFP8_E5M2_FORMAT_MAX).item()
    assert abs(scale8e5m2 - 2**-9) < 1e-12, f"E5M2 scale wrong: {scale8e5m2}"

    print(f"  ✓ Shared _compute_e8m0_scale: MXFP4={scale4}, E4M3={scale8e4m3}, E5M2={scale8e5m2:.6f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main runner (also works without pytest)
# ─────────────────────────────────────────────────────────────────────────────

def run_all():
    tests = [
        test_e4m3_format_max_is_448_not_240,
        test_e5m2_format_max_is_57344,
        test_e4m3_scale_formula,
        test_e5m2_scale_formula,
        test_e4m3_zero_passthrough,
        test_e5m2_zero_passthrough,
        test_e4m3_exact_level_identity,
        test_mxfp8_has_lower_mse_than_mxfp4,
        test_e4m3_residual_reduces_mse,
        test_e5m2_residual_reduces_mse,
        test_e4m3_adaptive_monotone,
        test_e5m2_adaptive_monotone,
        test_scale_code_path_is_shared,
    ]
    print("=" * 60)
    print("  MXFP8 Unit Tests")
    print("=" * 60)
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAIL {t.__name__}: {e}")
    print()
    print("=" * 60)
    if passed == len(tests):
        print(f"  ALL {passed} TESTS PASSED ✓")
    else:
        print(f"  {passed}/{len(tests)} passed — {len(tests)-passed} FAILED ✗")
    print("=" * 60)
    return passed == len(tests)


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
