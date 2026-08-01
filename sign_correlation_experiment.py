"""
sign_correlation_experiment.py
===============================
Single-layer targeted experiment on wav2vec2.encoder.layers.11.feed_forward.output_dense.

Measures sign(Δy_weight) · sign(Δy_act) — the error-cancellation correlation
coefficient ρ — to adjudicate between:
  (a) cancellation hypothesis:   ρ < -0.1  (weight and act errors offset each other)
  (b) null / structure hypothesis: |ρ| < 0.1 (no meaningful cancellation)
  (c) reinforcement (surprising): ρ > +0.1

Also reports: magnitude of each error term and Δy_act distribution shape
(skew, kurtosis) for mxfp4 vs mxfp4_residual_act_only at this layer.

Run:
    python sign_correlation_experiment.py

Output:
    results/sign_correlation_result.md
    results/sign_correlation_result.json
"""
from __future__ import annotations

import os
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats as sp_stats

# ── project imports ───────────────────────────────────────────────────────────
from core.layers import _quantise_weight, _quantise_activation
from frameworks.language import _resolve_modes

TARGET_LAYER = "wav2vec2.encoder.layers.11.feed_forward.output_dense"
N_SAMPLES    = 50
SEED         = 42
BLOCK_SIZE   = 32
MODEL_NAME   = "facebook/wav2vec2-base-960h"
OUT_DIR      = "results"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_linear(model, dotpath: str) -> nn.Linear:
    """Navigate dotpath to return the named Linear module."""
    parts = dotpath.split(".")
    m = model
    for p in parts:
        m = getattr(m, p)
    assert isinstance(m, nn.Linear), f"{dotpath} is not nn.Linear"
    return m


def _fake_linear(x: torch.Tensor, w: torch.Tensor, b, w_mode: str, a_mode: str) -> torch.Tensor:
    """Run a single Linear forward with independent weight/act quantization."""
    w_q, _ = _quantise_weight(w.float(), w_mode, BLOCK_SIZE)
    x_q, _ = _quantise_activation(x.float(), a_mode, BLOCK_SIZE)
    w_q = w_q.to(w.dtype)
    x_q = x_q.to(x.dtype)
    bias = b.to(x_q.dtype) if b is not None else None
    return F.linear(x_q, w_q, bias)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Target layer: {TARGET_LAYER}")
    print(f"Samples: {N_SAMPLES} from LibriSpeech test-clean (seed {SEED})")

    # ── Load model ────────────────────────────────────────────────────────────
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    print(f"\nLoading {MODEL_NAME}...")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model     = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.eval().to(device)

    target_module = _get_linear(model, TARGET_LAYER)
    W = target_module.weight.data.clone()   # [out, in]
    B = target_module.bias.data.clone() if target_module.bias is not None else None

    # ── Load calibration data — same first-N deterministic slice as run_diagnostics_extended ──
    from datasets import load_dataset
    print(f"Loading {N_SAMPLES} LibriSpeech test-clean samples...")
    ds = load_dataset("openslr/librispeech_asr", "clean", split="test",
                      streaming=True, trust_remote_code=True)
    samples = []
    for item in ds:
        samples.append(item["audio"]["array"])
        if len(samples) >= N_SAMPLES:
            break
    print(f"Loaded {len(samples)} samples.\n")

    # ── Hook to capture activations into this layer ───────────────────────────
    captured = {"x": None, "y_ref": None}

    def _hook(module, inp, out):
        captured["x"]     = inp[0].detach().float()
        captured["y_ref"] = out.detach().float()

    handle = target_module.register_forward_hook(_hook)

    # Accumulators
    # Per-sample ρ values for SE computation
    rho_per_sample = []
    mag_dw_list    = []    # mean |Δy_weight| per sample
    mag_da_list    = []    # mean |Δy_act|    per sample

    # For distribution shape: collect all Δy_act values under both modes
    delta_act_mxfp4     = []   # plain mxfp4 act errors
    delta_act_residual  = []   # mxfp4_residual_act_only act errors

    print("Running forward passes and computing per-sample statistics...")
    with torch.no_grad():
        for i, audio in enumerate(samples):
            inputs = processor(audio, sampling_rate=16000,
                               return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # BF16 reference forward
            model(**inputs)
            x_in  = captured["x"].cpu()       # [seq_len?, out_in_dim]
            y_ref = captured["y_ref"].cpu()    # same shape

            W_cpu = W.cpu(); B_cpu = B.cpu() if B is not None else None

            # ── Isolation pass 1: weight error only (w=mxfp4, a=bf16/fp32) ──
            # _resolve_modes("mxfp4") → ("mxfp4", "mxfp4")
            # We want w=mxfp4, a=fp32 (identity)
            y_w_only = _fake_linear(x_in, W_cpu, B_cpu,
                                    w_mode="mxfp4", a_mode="fp32")
            delta_weight = (y_w_only - y_ref).float()

            # ── Isolation pass 2: activation error only — residual act (w=fp32, a=mxfp4_residual) ──
            # This matches the "mxfp4_residual_act_only" mode: w stays BF16, acts use residual
            y_a_only = _fake_linear(x_in, W_cpu, B_cpu,
                                    w_mode="fp32", a_mode="mxfp4_residual")
            delta_act_res = (y_a_only - y_ref).float()

            # ── Isolation pass 3: activation error only — plain mxfp4 act (for shape comparison) ──
            y_a_plain = _fake_linear(x_in, W_cpu, B_cpu,
                                     w_mode="fp32", a_mode="mxfp4")
            delta_act_plain = (y_a_plain - y_ref).float()

            # ── Sign correlation ρ for this sample ───────────────────────────
            sign_w   = torch.sign(delta_weight).flatten()
            sign_a   = torch.sign(delta_act_res).flatten()
            rho_s    = (sign_w * sign_a).mean().item()
            rho_per_sample.append(rho_s)

            # ── Magnitudes ───────────────────────────────────────────────────
            mag_dw_list.append(delta_weight.abs().mean().item())
            mag_da_list.append(delta_act_res.abs().mean().item())

            # ── Collect Δy_act for distribution shape ─────────────────────
            delta_act_mxfp4.extend(delta_act_plain.flatten().tolist())
            delta_act_residual.extend(delta_act_res.flatten().tolist())

            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{N_SAMPLES} done...")

    handle.remove()

    # ── Aggregate statistics ──────────────────────────────────────────────────
    rho_arr = np.array(rho_per_sample)
    rho_mean = float(rho_arr.mean())
    rho_se   = float(rho_arr.std(ddof=1) / math.sqrt(len(rho_arr)))

    mag_dw_mean = float(np.mean(mag_dw_list))
    mag_da_mean = float(np.mean(mag_da_list))

    # Distribution shape (scipy)
    da_plain_arr = np.array(delta_act_mxfp4, dtype=np.float32)
    da_res_arr   = np.array(delta_act_residual, dtype=np.float32)

    skew_plain  = float(sp_stats.skew(da_plain_arr))
    skew_res    = float(sp_stats.skew(da_res_arr))
    kurt_plain  = float(sp_stats.kurtosis(da_plain_arr, fisher=True))   # excess kurtosis
    kurt_res    = float(sp_stats.kurtosis(da_res_arr, fisher=True))
    std_plain   = float(da_plain_arr.std())
    std_res     = float(da_res_arr.std())

    # ── Decision rule ─────────────────────────────────────────────────────────
    rho_lo = rho_mean - rho_se   # 1-SE confidence interval (not formal CI but practical)
    rho_hi = rho_mean + rho_se

    if rho_mean < -0.1 and rho_hi < 0.0:
        decision = (
            "CANCELLATION HYPOTHESIS SUPPORTED: ρ is significantly negative. "
            "Weight and activation errors offset each other at this layer; "
            "the residual activation pass disrupts this cancellation, "
            "unmasking weight error and worsening CTC log-probabilities."
        )
        outcome = "cancellation"
    elif abs(rho_mean) < 0.1 or (rho_lo < 0 and rho_hi > 0):
        decision = (
            "CANCELLATION HYPOTHESIS NOT SUPPORTED: ρ is within noise of zero. "
            "The WER regression is not explained by sign-correlated error cancellation. "
            "See distribution shape statistics below for alternative structural explanation."
        )
        outcome = "null"
    elif rho_mean > 0.1 and rho_lo > 0.0:
        decision = (
            "SURPRISING — POSITIVE CORRELATION: ρ is significantly positive. "
            "Weight and activation errors currently REINFORCE each other at this layer. "
            "This is a previously unconsidered explanation and should be flagged "
            "explicitly in the paper — it rules out both the cancellation hypothesis "
            "and the plain 'distribute error' story."
        )
        outcome = "reinforcement"
    else:
        decision = (
            "AMBIGUOUS: ρ is weakly outside the |0.1| threshold but the SE overlaps zero. "
            f"Effective sample size may be insufficient (n={N_SAMPLES}) to distinguish "
            "from noise. Do not assign to either hypothesis without additional samples."
        )
        outcome = "ambiguous"

    # ── Print ─────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print(f"  RESULT: {decision}")
    print("="*70)
    print(f"\n  ρ (sign correlation)  = {rho_mean:+.4f}  ±{rho_se:.4f} SE (n={N_SAMPLES} samples)")
    print(f"  mean |Δy_weight|      = {mag_dw_mean:.6f}")
    print(f"  mean |Δy_act_res|     = {mag_da_mean:.6f}  (mxfp4_residual_act_only)")
    print(f"\n  Δy_act distribution shape (this layer):")
    print(f"    plain mxfp4:       std={std_plain:.6f}  skew={skew_plain:+.4f}  excess_kurt={kurt_plain:+.4f}")
    print(f"    mxfp4_res_act_only: std={std_res:.6f}  skew={skew_res:+.4f}  excess_kurt={kurt_res:+.4f}")

    # ── Save markdown ─────────────────────────────────────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)
    md_path = os.path.join(OUT_DIR, "sign_correlation_result.md")
    with open(md_path, "w") as f:
        f.write("# Sign-Correlation Experiment — `layers.11.feed_forward.output_dense`\n\n")
        f.write(f"**CONCLUSION:** {decision}\n\n")
        f.write("---\n\n")
        f.write("## Evidence\n\n")
        f.write(f"| Statistic | Value |\n|---|---|\n")
        f.write(f"| ρ (mean sign-correlation) | `{rho_mean:+.4f}` |\n")
        f.write(f"| SE of ρ across {N_SAMPLES} samples | `{rho_se:.4f}` |\n")
        f.write(f"| 1-SE interval | `[{rho_lo:+.4f}, {rho_hi:+.4f}]` |\n")
        f.write(f"| Outcome | `{outcome}` |\n")
        f.write(f"| mean \\|Δy_weight\\| | `{mag_dw_mean:.6f}` |\n")
        f.write(f"| mean \\|Δy_act\\| (residual) | `{mag_da_mean:.6f}` |\n")
        f.write(f"| Dominant error term | `{'weight' if mag_dw_mean > mag_da_mean else 'activation'}` ({max(mag_dw_mean,mag_da_mean)/min(mag_dw_mean,mag_da_mean):.2f}× larger) |\n\n")
        f.write("## Δy_act Distribution Shape\n\n")
        f.write(f"| | std | skew | excess kurtosis |\n|---|---|---|---|\n")
        f.write(f"| plain mxfp4 | `{std_plain:.6f}` | `{skew_plain:+.4f}` | `{kurt_plain:+.4f}` |\n")
        f.write(f"| mxfp4_residual_act_only | `{std_res:.6f}` | `{skew_res:+.4f}` | `{kurt_res:+.4f}` |\n\n")
        f.write("## Experimental Protocol\n\n")
        f.write(f"- Model: `{MODEL_NAME}`\n")
        f.write(f"- Layer: `{TARGET_LAYER}`\n")
        f.write(f"- Samples: {N_SAMPLES} utterances from LibriSpeech test-clean (first {N_SAMPLES}, streaming)\n")
        f.write(f"- Isolation pass 1 (Δy_weight): w=mxfp4, a=fp32 (identity)\n")
        f.write(f"- Isolation pass 2 (Δy_act): w=fp32 (identity), a=mxfp4_residual\n")
        f.write(f"- Block size: {BLOCK_SIZE}\n")
        f.write(f"- Reference: BF16 full-precision forward\n")

    print(f"\n  Markdown → {md_path}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    json_path = os.path.join(OUT_DIR, "sign_correlation_result.json")
    with open(json_path, "w") as f:
        json.dump({
            "target_layer": TARGET_LAYER,
            "n_samples": N_SAMPLES,
            "rho_mean": rho_mean,
            "rho_se": rho_se,
            "rho_interval_1se": [rho_lo, rho_hi],
            "outcome": outcome,
            "decision": decision,
            "mag_delta_weight_mean": mag_dw_mean,
            "mag_delta_act_residual_mean": mag_da_mean,
            "delta_act_shape": {
                "plain_mxfp4":             {"std": std_plain, "skew": skew_plain, "excess_kurtosis": kurt_plain},
                "mxfp4_residual_act_only": {"std": std_res,   "skew": skew_res,   "excess_kurtosis": kurt_res},
            },
            "rho_per_sample": rho_per_sample,
        }, f, indent=2)
    print(f"  JSON     → {json_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
