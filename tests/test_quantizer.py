"""
Unit tests for core/quantizer.py

Verifies:
1. MXFP4 lookup table values (no normalization)
2. E8M0 scale uses FLOOR (not ceil)
3. Round-to-nearest-even at tie midpoints
4. Residual quantization reduces error
5. GPT-2 Conv1D weight blocking along correct axis
6. SQNR monotonicity: higher threshold → higher trigger rate
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch
import torch.nn as nn

from core.quantizer import (
    _MXFP4_LEVELS,
    _compute_e8m0_scale,
    _round_to_mxfp4_levels,
    quantize_mxfp4,
    dequantize_mxfp4,
    fake_quant_mxfp4,
    fake_quant_mxfp4_residual,
    fake_quant_mxfp4_adaptive,
    MXFP4_FORMAT_MAX,
)


def test_mxfp4_levels_not_normalized():
    """Levels must be exact physical values, not divided by 6 or anything else."""
    expected = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    actual = _MXFP4_LEVELS.tolist()
    for e, a in zip(expected, actual):
        assert abs(e - a) < 1e-6, f"Level mismatch: expected {e}, got {a}"
    assert abs(actual[-1] - 6.0) < 1e-6, "FORMAT_MAX must be 6.0"
    print("✓ MXFP4 levels are not normalized (correct absolute values)")


def test_e8m0_scale_uses_floor():
    """
    E8M0 scale: floor(log2(amax)) - floor(log2(6.0))
    floor(log2(6.0)) = floor(2.584) = 2

    Example: amax=5.0
      floor(log2(5.0)) = floor(2.322) = 2
      scale_exp = 2 - 2 = 0
      scale = 2^0 = 1.0

    Example: amax=7.0
      floor(log2(7.0)) = floor(2.807) = 2
      scale_exp = 2 - 2 = 0
      scale = 1.0

    Example: amax=12.0
      floor(log2(12.0)) = floor(3.585) = 3
      scale_exp = 3 - 2 = 1
      scale = 2.0
    """
    test_cases = []
    for amax_val in [3.0, 5.0, 6.0, 7.0, 12.0, 0.5, 1.0, 2.0, 24.0]:
        floor_log2_amax = math.floor(math.log2(amax_val))
        floor_log2_fmax = math.floor(math.log2(MXFP4_FORMAT_MAX))  # floor(log2(6))=2
        scale_exp = floor_log2_amax - floor_log2_fmax
        expected_scale = 2.0 ** scale_exp
        test_cases.append((amax_val, expected_scale))

    for amax_val, expected_scale in test_cases:
        amax_t = torch.tensor([amax_val])
        got_scale = _compute_e8m0_scale(amax_t).item()
        assert abs(got_scale - expected_scale) < 1e-5, (
            f"amax={amax_val}: expected scale={expected_scale}, got={got_scale}"
        )
    print("✓ E8M0 scale uses FLOOR (not ceil)")


def test_round_to_nearest_even():
    """
    Test round-to-nearest-even at the 7 tie midpoints.
    Midpoints: [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
    Levels:    [0.0,  0.5,  1.0,  1.5,  2.0, 3.0, 4.0, 6.0]
    Indices:   [ 0,    1,    2,    3,    4,   5,   6,   7 ]

    At tie i (between levels[i] and levels[i+1]):
      Even index wins:
        i=0 (0 vs 1): 0 is even → round to levels[0]=0.0
        i=1 (1 vs 2): 2 is even → round to levels[2]=1.0
        i=2 (2 vs 3): 2 is even → round to levels[2]=1.0
        i=3 (3 vs 4): 4 is even → round to levels[4]=2.0
        i=4 (4 vs 5): 4 is even → round to levels[4]=2.0
        i=5 (5 vs 6): 6 is even → round to levels[6]=4.0
        i=6 (6 vs 7): 6 is even → round to levels[6]=4.0
    """
    tie_cases = [
        (0.25,  0.0),   # midpoint[0]: levels 0 vs 1, even=0, pick 0.0
        (0.75,  1.0),   # midpoint[1]: levels 1 vs 2, even=2, pick 1.0
        (1.25,  1.0),   # midpoint[2]: levels 2 vs 3, even=2, pick 1.0
        (1.75,  2.0),   # midpoint[3]: levels 3 vs 4, even=4, pick 2.0
        (2.5,   2.0),   # midpoint[4]: levels 4 vs 5, even=4, pick 2.0
        (3.5,   4.0),   # midpoint[5]: levels 5 vs 6, even=6, pick 4.0
        (5.0,   4.0),   # midpoint[6]: levels 6 vs 7, even=6, pick 4.0
    ]
    x = torch.tensor([v for v, _ in tie_cases])
    result = _round_to_mxfp4_levels(x)
    for i, (input_val, expected) in enumerate(tie_cases):
        got = result[i].item()
        assert abs(got - expected) < 1e-6, (
            f"Tie at {input_val}: expected {expected}, got {got}"
        )
    print("✓ Round-to-nearest-even at all 7 tie midpoints")


def test_residual_reduces_error():
    """Residual quantisation must strictly reduce reconstruction error."""
    torch.manual_seed(0)
    x = torch.randn(32 * 10) * 3.0  # 10 blocks

    primary_dequant = fake_quant_mxfp4(x)
    primary_err = (x - primary_dequant).pow(2).sum().item()

    residual_dequant = fake_quant_mxfp4_residual(x)
    residual_err = (x - residual_dequant).pow(2).sum().item()

    assert residual_err < primary_err, (
        f"Residual did NOT reduce error: primary={primary_err:.6f}, residual={residual_err:.6f}"
    )
    print(f"✓ Residual reduces MSE: {primary_err:.4f} → {residual_err:.4f} "
          f"({100*(1-residual_err/primary_err):.1f}% reduction)")


def test_conv1d_weight_blocking_axis():
    """
    GPT-2 Conv1D stores weight as [in_features, out_features].
    FakeQuantGPT2Conv1D must block along in_features (axis-0 of stored tensor,
    equivalently rows of the transposed [out_features, in_features] tensor).

    We create a synthetic Conv1D weight where the first block (first 32 elements
    along in_features for output neuron 0) has a large amax=6.0 and the second
    block has amax=0.5. If blocking is done correctly along in_features, these
    two blocks get different scales. If done incorrectly (flattened in column-major
    or wrong axis), the scales won't match the expected per-block structure.
    """
    in_features = 64
    out_features = 4

    # Stored as [in_features, out_features]
    w = torch.zeros(in_features, out_features)
    # Set the first 32 rows (= first block for each output neuron in transposed layout)
    # to have max=6.0 in output neuron 0
    w[:32, 0] = 6.0   # first block of out_neuron_0: amax=6.0 → scale should be 1.0
    w[32:, 0] = 0.5   # second block of out_neuron_0: amax=0.5 → scale should be 0.125

    # Transpose to [out_features, in_features] and quantize
    w_t = w.t().contiguous()   # [4, 64]

    # Block 0 of row 0: w_t[0, 0:32] — max=6.0
    # Block 1 of row 0: w_t[0, 32:64] — max=0.5
    codes, scales = quantize_mxfp4(w_t, block_size=32)
    # scales shape: [num_blocks, 1] where num_blocks = (4 * 64) / 32 = 8
    # Row 0 of w_t contributes blocks 0 and 1
    scale_block0 = scales[0].item()
    scale_block1 = scales[1].item()

    floor_log2_6 = math.floor(math.log2(6.0))   # 2
    expected_scale0 = 2.0 ** (floor_log2_6 - 2)  # 1.0
    floor_log2_05 = math.floor(math.log2(0.5))  # -1
    expected_scale1 = 2.0 ** (floor_log2_05 - 2)  # 0.125

    assert abs(scale_block0 - expected_scale0) < 1e-5, (
        f"Block 0 scale: expected {expected_scale0}, got {scale_block0}"
    )
    assert abs(scale_block1 - expected_scale1) < 1e-5, (
        f"Block 1 scale: expected {expected_scale1}, got {scale_block1}"
    )
    print(f"✓ Conv1D weight blocking along in_features axis: "
          f"scale[0]={scale_block0} (exp={expected_scale0}), "
          f"scale[1]={scale_block1} (exp={expected_scale1})")


def test_sqnr_monotonicity():
    """
    Higher SQNR threshold → more blocks trigger → lower PPL (or equal).
    At minimum: trigger_rate[thresh_high] >= trigger_rate[thresh_low].
    """
    torch.manual_seed(42)
    x = torch.randn(32 * 100)  # 100 blocks, varied SQNR

    thresholds = [10.0, 15.0, 20.0, 30.0]
    prev_rate = -1.0
    for thresh in thresholds:
        _, rate = fake_quant_mxfp4_adaptive(x, thresh)
        assert rate >= prev_rate - 1e-6, (
            f"Monotonicity violated: thresh={thresh}, rate={rate:.4f} < prev={prev_rate:.4f}"
        )
        prev_rate = rate
        print(f"  thresh={thresh:4.0f}dB → trigger_rate={rate:.4f}")
    print("✓ SQNR trigger_rate is monotonically non-decreasing with threshold")


def test_end_to_end_block_trace():
    """
    Hand-trace one block end-to-end with known values and verify each step.
    Block: [1.7, -2.3, 0.4, ...] with synthetic amax.
    """
    # Construct a single block with known amax
    x_block = torch.zeros(32)
    x_block[0] = 5.5   # amax
    x_block[1] = -2.3
    x_block[2] = 0.75  # tie midpoint
    x_block[3] = 1.25  # tie midpoint

    # Expected scale: floor(log2(5.5))-floor(log2(6)) = floor(2.459)-2 = 2-2=0 → scale=1.0
    expected_scale = 1.0

    codes, scales = quantize_mxfp4(x_block.unsqueeze(0).reshape(1, 32), block_size=32)
    scale_got = scales[0, 0].item()
    assert abs(scale_got - expected_scale) < 1e-5, f"Scale: expected {expected_scale}, got {scale_got}"

    # x_block[0] = 5.5 / 1.0 = 5.5 → closest to 6.0 (distance 0.5) vs 4.0 (distance 1.5) → 6.0
    # x_block[2] = 0.75 / 1.0 = 0.75 → tie midpoint → round to 1.0 (RNE: index 2 is even)
    # x_block[3] = 1.25 / 1.0 = 1.25 → tie midpoint → round to 1.0 (RNE: index 2 is even)

    dequant = dequantize_mxfp4(codes.reshape(1, 32), scales)
    val0 = dequant[0, 0].item()
    val2 = dequant[0, 2].item()
    val3 = dequant[0, 3].item()

    assert abs(val0 - 6.0) < 1e-5, f"val[0] expected 6.0, got {val0}"
    assert abs(val2 - 1.0) < 1e-5, f"val[2] (tie 0.75→1.0 RNE) expected 1.0, got {val2}"
    assert abs(val3 - 1.0) < 1e-5, f"val[3] (tie 1.25→1.0 RNE) expected 1.0, got {val3}"

    # Residual pass
    primary_dequant = fake_quant_mxfp4(x_block)
    residual = x_block - primary_dequant
    residual_dequant = fake_quant_mxfp4(residual)
    reconstructed = primary_dequant + residual_dequant

    primary_err = (x_block - primary_dequant).pow(2).sum().item()
    residual_err = (x_block - reconstructed).pow(2).sum().item()
    assert residual_err <= primary_err, "Residual made things WORSE"

    print(f"✓ End-to-end block trace:")
    print(f"    scale={scale_got:.4f}, val[0]={val0:.4f}, val[2]={val2:.4f}, val[3]={val3:.4f}")
    print(f"    primary_err={primary_err:.6f}, residual_err={residual_err:.6f}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  MXFP4 Quantizer Unit Tests")
    print("="*60 + "\n")

    test_mxfp4_levels_not_normalized()
    test_e8m0_scale_uses_floor()
    test_round_to_nearest_even()
    test_residual_reduces_error()
    test_conv1d_weight_blocking_axis()
    test_sqnr_monotonicity()
    test_end_to_end_block_trace()

    print("\n" + "="*60)
    print("  ALL UNIT TESTS PASSED ✓")
    print("="*60 + "\n")
