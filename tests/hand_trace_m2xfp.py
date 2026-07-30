"""
tests/hand_trace_m2xfp.py
=========================
Hand-trace one real 32-element block through both M2XFP paths,
printing every intermediate value so the implementation can be
verified against the paper (Sec 4.4, Algorithm 1, Eq. 4).

Run:
    python tests/hand_trace_m2xfp.py

No dependencies beyond torch and this project's core module.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch

# ── Import our implementation ────────────────────────────────────────────────
from core.quantizer import (
    _MXFP4_LEVELS,
    _FP6_E2M3_LEVELS,
    MXFP4_FORMAT_MAX,
    M2XFP_FP6_FORMAT_MAX,
    _M2XFP_RATIOS,
    _M2XFP_BIAS_RANGE,
    _compute_e8m0_scale,
    _round_to_mxfp4_levels,
    fake_quant_m2xfp_weight,
    fake_quant_m2xfp_act,
)

BLOCK_SIZE = 32
SUB_GROUP_SIZE = 8
SEP = "─" * 72

# ─────────────────────────────────────────────────────────────────────────────
# Shared test block: 32 elements chosen to exercise interesting cases
#   - a large positive outlier at index 0
#   - a large negative outlier at index 8 (start of subgroup 1)
#   - several zeros
#   - tie candidates in subgroup 2 (indices 16-23)
# ─────────────────────────────────────────────────────────────────────────────
torch.manual_seed(0)
x = torch.tensor([
    # Subgroup 0 (indices 0-7)
     5.7,  1.2, -0.5,  0.0,  2.3, -1.8,  0.9,  3.1,
    # Subgroup 1 (indices 8-15)
    -5.9,  0.4,  0.0, -2.2,  1.5,  0.7, -0.3,  4.0,
    # Subgroup 2 (indices 16-23) — two values that tie at FP4 magnitude 3.0
     3.0, -3.0,  0.1, -0.1,  2.8, -2.8,  0.5, -0.5,
    # Subgroup 3 (indices 24-31)
     0.0,  0.0,  1.0, -1.0,  0.25, 0.75, -0.25, -0.75,
], dtype=torch.float32)

assert x.shape[0] == BLOCK_SIZE

print(f"\n{'='*72}")
print(" M2XFP Hand-Trace — one 32-element block")
print(f"{'='*72}")
print(f"\nInput block:\n  {x.tolist()}")


# ─────────────────────────────────────────────────────────────────────────────
# PART 1: WEIGHT PATH — Sg-EM-2bit
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print(" PART 1 — WEIGHT PATH: Sg-EM-2bit (Subgroup-scale Extra Mantissa)")
print(SEP)

x_flat = x.unsqueeze(0).float()        # [1, 32]
amax = torch.amax(x_flat.abs(), dim=-1, keepdim=True).clamp(min=1e-8)
exp_base = torch.floor(torch.log2(amax)) - math.floor(math.log2(MXFP4_FORMAT_MAX))
scale_0 = torch.pow(2.0, exp_base)     # bias=0 scale

print(f"\n  Group amax            = {amax.item():.4f}")
print(f"  E8M0 exponent (base)  = {exp_base.item():.0f}")
print(f"  E8M0 scale (bias=0)   = {scale_0.item():.6f}")
print(f"  FP4 FORMAT_MAX        = {MXFP4_FORMAT_MAX}")

sub_groups_per_group = BLOCK_SIZE // SUB_GROUP_SIZE  # 4

for b in _M2XFP_BIAS_RANGE:
    scale_b = torch.pow(2.0, exp_base + b)            # [1, 1]
    print(f"\n  ── Bias b={b:+d}  →  group scale = {scale_b.item():.6f}")

    x_sg = x_flat.reshape(sub_groups_per_group, SUB_GROUP_SIZE)   # [4, 8]
    scale_b_sg = scale_b.expand(1, sub_groups_per_group).reshape(
        sub_groups_per_group, 1
    )                                                              # [4, 1]

    best_mses = []
    best_ks = []
    best_dqs = []

    for sg_i in range(sub_groups_per_group):
        sg = x_sg[sg_i]                 # [8]
        s  = scale_b_sg[sg_i, 0]        # scalar
        best_mse = float('inf')
        best_k_val = -1
        best_dq_sg = None

        for k, ratio in enumerate(_M2XFP_RATIOS.tolist()):
            eff_scale = s * ratio
            q_abs = _round_to_mxfp4_levels((sg.abs() / eff_scale).clamp(0, MXFP4_FORMAT_MAX))
            dq = q_abs * sg.sign() * eff_scale
            mse = ((dq - sg) ** 2).mean().item()
            if sg_i < 2:  # print detail for first two subgroups
                print(f"       SG{sg_i}  k={k} ratio={ratio:.2f}  eff_scale={eff_scale:.5f}  "
                      f"MSE={mse:.6f}")
            if mse < best_mse:
                best_mse = mse
                best_k_val = k
                best_dq_sg = dq

        best_mses.append(best_mse)
        best_ks.append(best_k_val)
        best_dqs.append(best_dq_sg)
        print(f"       SG{sg_i}  → best k*={best_k_val} "
              f"ratio={_M2XFP_RATIOS[best_k_val].item():.2f}  MSE={best_mse:.6f}")

    group_mse = sum(best_mses) / sub_groups_per_group
    print(f"    Group MSE (avg subgroups) = {group_mse:.6f}")

print(f"\n  Calling fake_quant_m2xfp_weight ...")
w_out = fake_quant_m2xfp_weight(x, block_size=BLOCK_SIZE, sub_group_size=SUB_GROUP_SIZE)
w_nmse = ((w_out - x) ** 2).mean() / (x ** 2).mean()
w_sqnr = 10 * math.log10(((x ** 2).mean() / ((w_out - x) ** 2).mean()).item())
print(f"  Output (weight path):\n    {w_out.tolist()}")
print(f"  NMSE  = {w_nmse.item():.6f}")
print(f"  SQNR  = {w_sqnr:.2f} dB")


# ─────────────────────────────────────────────────────────────────────────────
# PART 2: ACTIVATION PATH — Elem-EM-top1
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print(" PART 2 — ACTIVATION PATH: Elem-EM-top1 (Element-level Extra Mantissa)")
print(SEP)

x_flat = x.unsqueeze(0).float()        # [1, 32]
amax = torch.amax(x_flat.abs(), dim=-1, keepdim=True).clamp(min=1e-8)
scale = _compute_e8m0_scale(amax)      # [1, 1]

print(f"\n  Group amax   = {amax.item():.4f}")
print(f"  E8M0 scale   = {scale.item():.6f}")

# Step 2: FP4 quantize all
x_scaled = x_flat / scale
fp4_abs = _round_to_mxfp4_levels(x_scaled.abs().clamp(0, MXFP4_FORMAT_MAX))
fp4_vals = fp4_abs * x_scaled.sign()
print(f"\n  FP4 quantized (scaled domain):\n    {fp4_vals.squeeze().tolist()}")
print(f"  FP4 dequantized:\n    {(fp4_vals * scale).squeeze().tolist()}")

# Per-subgroup top-1 + FP6 promotion
fp4_levels = _MXFP4_LEVELS
fp6_levels = _FP6_E2M3_LEVELS

x_sg = x_flat.reshape(sub_groups_per_group, SUB_GROUP_SIZE)          # [4, 8]
fp4_sg = fp4_abs.reshape(sub_groups_per_group, SUB_GROUP_SIZE)        # [4, 8]
sign_sg = x_flat.sign().reshape(sub_groups_per_group, SUB_GROUP_SIZE) # [4, 8]
scale_sg = scale.expand(1, sub_groups_per_group).reshape(
    sub_groups_per_group, 1
)                                                                      # [4, 1]

print(f"\n  Per-subgroup top-1 selection and FP6 promotion:")
for sg_i in range(sub_groups_per_group):
    fp4_abs_sg = fp4_sg[sg_i]
    orig_sg    = x_sg[sg_i]
    s          = scale_sg[sg_i, 0]

    # Top-1: largest FP4 magnitude, lowest index on tie
    top1_val, top1_idx = torch.topk(fp4_abs_sg, k=1, largest=True, sorted=False)
    top1_idx = top1_idx.item()
    top1_fp4_abs = fp4_abs_sg[top1_idx].item()

    # Bias-clamp FP6 encoding
    orig_scaled_abs = (orig_sg[top1_idx] / s).abs().clamp(0.0, M2XFP_FP6_FORMAT_MAX)

    fp4_idx = torch.searchsorted(
        fp4_levels.contiguous(), torch.tensor(top1_fp4_abs)
    ).clamp(0, len(fp4_levels) - 1).item()

    fp6_idx_raw = torch.searchsorted(
        fp6_levels.contiguous(), torch.tensor(orig_scaled_abs.item())
    ).clamp(0, len(fp6_levels) - 1).item()  # searchsorted returns N on overflow; clamp to N-1

    # Bias-clamp range: FP6 index must have same top-4 bits as FP4 index.
    # fp4_idx*4 is the base FP6 index for the same E2M1 level.
    # Range is [fp4_idx*4 - 1, fp4_idx*4 + 2], BUT both bounds must stay in [0, 31].
    # (Since FP4 max index is 7, fp4_idx*4 + 2 is 30, which fits in 31).
    fp6_idx_min = max(0, min(len(fp6_levels) - 1, fp4_idx * 4 - 1))
    fp6_idx_max = max(0, min(len(fp6_levels) - 1, fp4_idx * 4 + 2))
    fp6_idx = max(fp6_idx_min, min(fp6_idx_max, fp6_idx_raw))
    fp6_mag = fp6_levels[fp6_idx].item()

    fp6_dq = fp6_mag * sign_sg[sg_i, top1_idx].item() * s.item()

    print(f"\n  SG{sg_i}  fp4_abs = {fp4_abs_sg.tolist()}")
    print(f"       top1_idx      = {top1_idx}  (value {top1_fp4_abs:.4f} in scaled domain)")
    print(f"       orig_scaled   = {orig_scaled_abs.item():.6f}")
    print(f"       fp4_idx       = {fp4_idx}  → fp4_level = {fp4_levels[fp4_idx].item():.4f}")
    print(f"       fp6_idx_raw   = {fp6_idx_raw}  → clamp range [{fp6_idx_min}, {fp6_idx_max}]  → fp6_idx = {fp6_idx}")
    print(f"       fp6_mag       = {fp6_mag:.6f}  (scaled domain)")
    print(f"       fp6 dequant   = {fp6_dq:.6f}  (original domain)")
    print(f"       2-bit metadata: fp6_idx - fp4_idx*4 + 1 = {fp6_idx - fp4_idx*4 + 1}  "
          f"→ binary {bin((fp6_idx - fp4_idx*4 + 1) & 0x3)[2:].zfill(2)}")

print(f"\n  Calling fake_quant_m2xfp_act ...")
a_out = fake_quant_m2xfp_act(x, block_size=BLOCK_SIZE, sub_group_size=SUB_GROUP_SIZE)
a_nmse = ((a_out - x) ** 2).mean() / (x ** 2).mean()
a_sqnr = 10 * math.log10(((x ** 2).mean() / ((a_out - x) ** 2).mean()).item())
print(f"  Output (act path):\n    {a_out.tolist()}")
print(f"  NMSE  = {a_nmse.item():.6f}")
print(f"  SQNR  = {a_sqnr:.2f} dB")


# ─────────────────────────────────────────────────────────────────────────────
# PART 3: EBW accounting (Eq. 2)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print(" PART 3 — Effective Bit Width (EBW) Accounting  [Eq. 2 of paper]")
print(SEP)

k = BLOCK_SIZE         # group size
B_elem = 4             # FP4 = 4 bits per element
B_meta = 2 * (k // SUB_GROUP_SIZE)   # 2 bits per subgroup × 4 subgroups = 8
B_scale = 8            # E8M0

EBW = B_elem + (B_meta + B_scale) / k
print(f"\n  group_size     k  = {k}")
print(f"  sub_group_size    = {SUB_GROUP_SIZE}")
print(f"  B_elem            = {B_elem}   (FP4 element bits)")
print(f"  B_meta            = {B_meta}   (2 bits/subgroup × {k//SUB_GROUP_SIZE} subgroups)")
print(f"  B_scale           = {B_scale}   (E8M0 shared scale)")
print(f"  EBW = B_elem + (B_meta + B_scale) / k")
print(f"      = {B_elem} + ({B_meta} + {B_scale}) / {k}")
print(f"      = {B_elem} + {(B_meta + B_scale) / k:.4f}")
print(f"      = {EBW:.4f}  bits/value")
print(f"\n  ✓ Paper claims ~4.5 bits/value → computed {EBW:.1f}  ✓")

from core.quantizer import bits_per_value
print(f"  bits_per_value('m2xfp') = {bits_per_value('m2xfp')}")


# ─────────────────────────────────────────────────────────────────────────────
# PART 4: Baseline comparison
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print(" PART 4 — Sanity: Compare M2XFP vs plain MXFP4 on this block")
print(SEP)

from core.quantizer import fake_quant_mxfp4
mxfp4_out = fake_quant_mxfp4(x, block_size=BLOCK_SIZE)
mxfp4_nmse = ((mxfp4_out - x) ** 2).mean() / (x ** 2).mean()
mxfp4_sqnr = 10 * math.log10(((x ** 2).mean() / ((mxfp4_out - x) ** 2).mean()).item())

print(f"\n  {'Format':<30} {'NMSE':>12}  {'SQNR (dB)':>12}")
print(f"  {'─'*30} {'─'*12}  {'─'*12}")
print(f"  {'Plain MXFP4 (4.25 bpv)':<30} {mxfp4_nmse.item():>12.6f}  {mxfp4_sqnr:>12.2f}")
print(f"  {'M2XFP weight (Sg-EM)':<30} {w_nmse.item():>12.6f}  {w_sqnr:>12.2f}")
print(f"  {'M2XFP act (Elem-EM)':<30} {a_nmse.item():>12.6f}  {a_sqnr:>12.2f}")

if w_nmse.item() < mxfp4_nmse.item():
    print(f"\n  ✓ Weight path NMSE is LOWER than plain MXFP4 — expected.")
else:
    print(f"\n  ✗ WARNING: Weight path NMSE is NOT better than plain MXFP4. Check implementation!")

if a_nmse.item() < mxfp4_nmse.item():
    print(f"  ✓ Activation path NMSE is LOWER than plain MXFP4 — expected.")
else:
    print(f"  ✗ WARNING: Activation path NMSE is NOT better than plain MXFP4. Check implementation!")

print(f"\n{'='*72}")
print(" Hand-trace complete.")
print(f"{'='*72}\n")
