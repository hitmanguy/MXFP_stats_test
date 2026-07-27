"""
run_speech_sweep.py
===================
End-to-end speech quantisation sweep for Wav2Vec2 / LibriSpeech test-clean.

Covers Steps 2–5 from the task spec:
  Step 2: FP32 baseline vs. documented WER (3.4%)
  Step 3: Ablation — lm_head sensitivity, feature-extractor sensitivity
  Step 4: Naive MXFP4 / MXFP8 sweep
  Step 5: Residual and adaptive SQNR sweep + monotonicity check

Usage:
    python run_speech_sweep.py [--n-samples N] [--device cpu|cuda] [--steps 2,3,4,5]

Defaults:
    --n-samples 200  (200 utterances, ~10 min on CPU, ~2 min on GPU)
    --device    auto
    --steps     2,3,4,5
"""
from __future__ import annotations

# ── SSL cert fix (must be before any network-aware library imports) ──
import os as _os
_os.environ["HF_HOME"] = r"D:\hf_cache"
_os.environ["HF_DATASETS_CACHE"] = r"D:\hf_cache\datasets"
_os.environ["TRANSFORMERS_CACHE"] = r"D:\hf_cache\transformers"

try:
    import certifi as _certifi
    _os.environ.setdefault("SSL_CERT_FILE", _certifi.where())
    _os.environ.setdefault("REQUESTS_CA_BUNDLE", _certifi.where())
except ImportError:
    pass
# ────────────────────────────────────────────────────────────────────────────

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

DOCUMENTED_WER = 3.4   # % — HF model card, facebook/wav2vec2-base-960h, greedy, test-clean
WER_BASELINE_TOLERANCE = 1.5  # pp tolerance: baseline within 1.5 pp is acceptable


def _harness(
    quant_mode: str,
    n_samples: int,
    device: torch.device,
    seed: int = 42,
    skip_feature_extractor: bool = False,
    extra_skip_names: Optional[List[str]] = None,
) -> Dict:
    from frameworks.speech import SpeechEvalHarness
    h = SpeechEvalHarness(
        model_name="facebook/wav2vec2-base-960h",
        quant_mode=quant_mode,
        n_samples=n_samples,
        seed=seed,
        device=device,
        skip_feature_extractor=skip_feature_extractor,
        extra_skip_names=extra_skip_names or [],
    )
    return h.run()


def _fmt(label: str, wer_pct: float, ref: Optional[float] = None) -> str:
    delta_str = ""
    if ref is not None:
        delta = wer_pct - ref
        delta_str = f"  Δref={delta:+.2f} pp"
    return f"  {label:<50} WER={wer_pct:6.2f}%{delta_str}"


def _save_results(results: List[Dict], out_path: Path) -> None:
    with open(out_path, "w") as f:
        for r in results:
            # Only write JSON-serialisable scalar fields
            row = {k: v for k, v in r.items() if k not in ("references", "hypotheses")}
            f.write(json.dumps(row) + "\n")
    print(f"\n  [Saved] {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — FP32 Baseline
# ─────────────────────────────────────────────────────────────────────────────

def step2_baseline(n_samples: int, device: torch.device) -> Dict:
    print("\n" + "="*70)
    print("  STEP 2 — FP32 Baseline Sanity Check")
    print(f"  Documented reference (greedy, test-clean): {DOCUMENTED_WER:.1f}%")
    print("="*70)

    result = _harness("fp32", n_samples, device)
    wer_pct = result["wer_pct"]
    delta = wer_pct - DOCUMENTED_WER

    print("\n  ┌──────────────────────────────────────────────────────────────┐")
    print(f"  │  FP32 Baseline WER:  {wer_pct:.2f}%                               │")
    print(f"  │  Documented WER:     {DOCUMENTED_WER:.1f}%                                │")
    print(f"  │  Δ from reference:   {delta:+.2f} pp                              │")

    if abs(delta) <= WER_BASELINE_TOLERANCE:
        print( "  │  Status: [OK] BASELINE OK - within tolerance                    │")
    else:
        print( "  │  Status: [X] BASELINE OFF - check audio preprocessing!         │")
        print(f"  │  (tolerance is ±{WER_BASELINE_TOLERANCE} pp)                                    │")

    print("  └──────────────────────────────────────────────────────────────┘\n")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Sensitivity Ablation
# ─────────────────────────────────────────────────────────────────────────────

def step3_ablation(n_samples: int, device: torch.device, baseline_wer: float) -> List[Dict]:
    """
    Four ablation conditions:
      A) Quantise everything EXCEPT lm_head (skip lm_head)
      B) Quantise everything INCLUDING lm_head  (skip nothing extra)
      C) Quantise everything EXCEPT feature extractor conv layers
      D) Quantise everything INCLUDING feature extractor conv layers

    All use MXFP4 as the "stress test" format (most aggressive).
    """
    print("\n" + "="*70)
    print("  STEP 3 — Sensitivity Ablation (MXFP4, analogous to lm_head check)")
    print("="*70)

    conditions = [
        # (label, skip_feature_extractor, extra_skip_names)
        ("A: MXFP4, skip lm_head (default policy)",      False, ["lm_head"]),
        ("B: MXFP4, include lm_head (no skip)",           False, []),
        ("C: MXFP4, skip feature extractor convs",        True,  ["lm_head"]),
        ("D: MXFP4, include feature extractor convs",     False, ["lm_head"]),
    ]

    results = []
    print(f"\n  FP32 baseline WER = {baseline_wer:.2f}%\n")

    for label, skip_fe, extra_skip in conditions:
        print(f"\n  ── Running: {label} ──")
        r = _harness(
            quant_mode="mxfp4",
            n_samples=n_samples,
            device=device,
            skip_feature_extractor=skip_fe,
            extra_skip_names=extra_skip,
        )
        r["ablation_label"] = label
        r["baseline_wer"] = baseline_wer
        results.append(r)

    # Print ablation table
    print("\n\n  ABLATION TABLE  (MXFP4, all conditions)")
    print("  " + "-"*72)
    print(f"  {'Condition':<50} {'WER %':>7}  {'Δ baseline':>10}  {'Δ vs A':>9}")
    print("  " + "-"*72)

    wer_A = None
    for r in results:
        w = r["wer_pct"]
        lbl = r["ablation_label"]
        delta_base = w - baseline_wer
        if wer_A is None:
            wer_A = w
        delta_A = w - wer_A
        print(f"  {lbl:<50} {w:>7.2f}%  {delta_base:>+9.2f}pp  {delta_A:>+8.2f}pp")

    print("  " + "-"*72)

    # Assess: is lm_head disproportionately sensitive?
    wer_skip_lm  = results[0]["wer_pct"]  # A: skip lm_head
    wer_incl_lm  = results[1]["wer_pct"]  # B: include lm_head
    lm_impact = wer_incl_lm - wer_skip_lm

    wer_skip_fe  = results[2]["wer_pct"]  # C: skip FE
    wer_incl_fe  = results[3]["wer_pct"]  # D: include FE  (same as A but called out)
    fe_impact = wer_incl_fe - wer_skip_fe  # should be close to 0 since D uses skip_fe=False

    print(f"\n  lm_head sensitivity:       {lm_impact:+.2f} pp  (B − A)")
    print(f"  Feature extractor impact:  {fe_impact:+.2f} pp  (D − C)")

    # Decision: skip if adding that layer causes > 10 pp blowup
    BLOWUP_THRESHOLD = 10.0   # pp

    print("\n  Skip policy determination:")
    if lm_impact > BLOWUP_THRESHOLD:
        print(f"  [X] lm_head: SKIP in default config (impact {lm_impact:+.2f} pp > {BLOWUP_THRESHOLD} pp threshold)")
    else:
        print(f"  [OK] lm_head: safe to quantise (impact {lm_impact:+.2f} pp <= {BLOWUP_THRESHOLD} pp threshold)")

    if abs(fe_impact) > BLOWUP_THRESHOLD:
        print(f"  [X] Feature extractor: SKIP in default config (impact {fe_impact:+.2f} pp > {BLOWUP_THRESHOLD} pp threshold)")
    else:
        print(f"  [OK] Feature extractor: safe to quantise (impact {fe_impact:+.2f} pp <= {BLOWUP_THRESHOLD} pp threshold)\n")

    return results

# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Naive MXFP4 / MXFP8 sweep
# ─────────────────────────────────────────────────────────────────────────────

def step4_naive_sweep(n_samples: int, device: torch.device, baseline_wer: float) -> List[Dict]:
    print("\n" + "="*70)
    print("  STEP 4 — Naive MXFP4 / MXFP8 Sweep")
    print(f"  (Default skip policy: lm_head skipped, feature extractor included)")
    print("="*70)

    modes = [
        "mxfp4",
        "mxfp8_e4m3",
        "mxfp8_e5m2",
    ]

    results = []
    for mode in modes:
        print(f"\n  ── Running: {mode} ──")
        r = _harness(quant_mode=mode, n_samples=n_samples, device=device)
        results.append(r)

    # Print table
    from core.quantizer import bits_per_value
    print("\n\n  NAIVE SWEEP TABLE")
    print("  " + "-"*65)
    print(f"  {'Mode':<25} {'eff_bits':>8}  {'WER %':>7}  {'Δ baseline':>10}")
    print("  " + "-"*65)
    print(f"  {'fp32 (baseline)':<25} {'32':>8}  {baseline_wer:>7.2f}%  {'—':>10}")
    for r in results:
        w = r["wer_pct"]
        m = r["quant_mode"]
        bits = bits_per_value(m)
        delta = w - baseline_wer
        print(f"  {m:<25} {bits:>8.1f}  {w:>7.2f}%  {delta:>+9.2f}pp")
    print("  " + "-"*65)

    # Sanity check: MXFP8 should be better than MXFP4
    wer_mxfp4 = next(r["wer_pct"] for r in results if r["quant_mode"] == "mxfp4")
    wer_mxfp8_e4m3 = next(r["wer_pct"] for r in results if r["quant_mode"] == "mxfp8_e4m3")
    wer_mxfp8_e5m2 = next(r["wer_pct"] for r in results if r["quant_mode"] == "mxfp8_e5m2")

    print("\n  Cross-format sanity checks:")
    ok1 = wer_mxfp8_e4m3 < wer_mxfp4
    ok2 = wer_mxfp8_e5m2 < wer_mxfp4
    print(f"  MXFP8 E4M3 ({wer_mxfp8_e4m3:.2f}%) < MXFP4 ({wer_mxfp4:.2f}%): {'[OK] PASS' if ok1 else '[X] FAIL'}")
    print(f"  MXFP8 E5M2 ({wer_mxfp8_e5m2:.2f}%) < MXFP4 ({wer_mxfp4:.2f}%): {'[OK] PASS' if ok2 else '[X] FAIL'}\n")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Residual + Adaptive SQNR sweep
# ─────────────────────────────────────────────────────────────────────────────

def step5_residual_adaptive(n_samples: int, device: torch.device, baseline_wer: float) -> List[Dict]:
    print("\n" + "="*70)
    print("  STEP 5 — Residual + Adaptive SQNR Sweep")
    print("="*70)

    modes = [
        # Residual
        "mxfp4_residual",
        "mxfp8_e4m3_residual",
        "mxfp8_e5m2_residual",
        # Adaptive (3 threshold points for monotonicity check)
        "mxfp4_adaptive_12",
        "mxfp4_adaptive_15",
        "mxfp4_adaptive_18",
        # MXFP4 naive (reference for monotonicity anchor)
        "mxfp4",
    ]

    results = []
    for mode in modes:
        print(f"\n  ── Running: {mode} ──")
        r = _harness(quant_mode=mode, n_samples=n_samples, device=device)
        results.append(r)

    from core.quantizer import bits_per_value

    print("\n\n  RESIDUAL + ADAPTIVE TABLE")
    print("  " + "-"*65)
    print(f"  {'Mode':<30} {'eff_bits':>8}  {'WER %':>7}  {'Δ baseline':>10}")
    print("  " + "-"*65)
    print(f"  {'fp32 (baseline)':<30} {'32':>8}  {baseline_wer:>7.2f}%  {'—':>10}")
    for r in results:
        w = r["wer_pct"]
        m = r["quant_mode"]
        bits = bits_per_value(m)
        delta = w - baseline_wer
        print(f"  {m:<30} {bits:>8.1f}  {w:>7.2f}%  {delta:>+9.2f}pp")
    print("  " + "-"*65)

    # Monotonicity check: as SQNR threshold increases (more residual used),
    # WER must never get worse. Higher threshold = more blocks use residual.
    print("\n  SQNR Monotonicity Check (WER must not increase as threshold rises):")
    adaptive_thresholds = [12, 15, 18]
    adaptive_wers = {}
    for t in adaptive_thresholds:
        mode = f"mxfp4_adaptive_{t}"
        r_match = next((r for r in results if r["quant_mode"] == mode), None)
        if r_match is not None:
            adaptive_wers[t] = r_match["wer_pct"]

    prev_thresh = None
    prev_wer = None
    all_mono = True
    MONO_TOLERANCE = 0.5   # pp — small floating-point tolerance
    for t in sorted(adaptive_wers.keys()):
        w = adaptive_wers[t]
        if prev_wer is not None:
            # Higher threshold → more residual triggered → WER should not get worse
            mono_ok = w <= prev_wer + MONO_TOLERANCE
            sym = "[OK]" if mono_ok else "[X] VIOLATION!"
            if not mono_ok:
                all_mono = False
            print(f"    thresh={prev_thresh} ({prev_wer:.2f}%) → thresh={t} ({w:.2f}%):  {sym}")
        prev_thresh, prev_wer = t, w

    if all_mono:
        print("\n  [OK] Monotonicity property holds across all threshold points.\n")
    else:
        print("\n  [X] Monotonicity violated - investigate adaptive quantiser.\n")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Wav2Vec2 speech quantisation sweep")
    parser.add_argument("--n-samples", type=int, default=200,
                        help="Number of LibriSpeech utterances (default 200)")
    parser.add_argument("--device", default="auto",
                        help="Device: cpu, cuda, or auto (default: auto)")
    parser.add_argument("--steps", default="2,3,4,5",
                        help="Comma-separated steps to run (default: 2,3,4,5)")
    parser.add_argument("--out-dir", default="results",
                        help="Output directory for JSONL results (default: results/)")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    steps = set(int(s.strip()) for s in args.steps.split(","))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[speech_sweep] n_samples={args.n_samples}, device={device}, steps={sorted(steps)}")
    print(f"[speech_sweep] Documented reference WER = {DOCUMENTED_WER}% (greedy, test-clean)")

    all_results: List[Dict] = []
    baseline_wer = DOCUMENTED_WER  # fallback if step 2 is skipped

    # ── Step 2: Baseline ──────────────────────────────────────────────────────
    if 2 in steps:
        r_base = step2_baseline(args.n_samples, device)
        all_results.append(r_base)
        baseline_wer = r_base["wer_pct"]
        _save_results([r_base], out_dir / "speech_step2_baseline.jsonl")

    # ── Step 3: Ablation ──────────────────────────────────────────────────────
    if 3 in steps:
        r_ablation = step3_ablation(args.n_samples, device, baseline_wer)
        all_results.extend(r_ablation)
        _save_results(r_ablation, out_dir / "speech_step3_ablation.jsonl")

    # ── Step 4: Naive sweep ───────────────────────────────────────────────────
    if 4 in steps:
        r_naive = step4_naive_sweep(args.n_samples, device, baseline_wer)
        all_results.extend(r_naive)
        _save_results(r_naive, out_dir / "speech_step4_naive.jsonl")

    # ── Step 5: Residual + adaptive ───────────────────────────────────────────
    if 5 in steps:
        r_resid = step5_residual_adaptive(args.n_samples, device, baseline_wer)
        all_results.extend(r_resid)
        _save_results(r_resid, out_dir / "speech_step5_residual.jsonl")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  STEP 6 CHECKPOINT SUMMARY")
    print("="*70)
    print(f"  Documented WER (reference): {DOCUMENTED_WER:.1f}%")
    print(f"  FP32 baseline WER:          {baseline_wer:.2f}%")
    print(f"  Δ from reference:           {baseline_wer - DOCUMENTED_WER:+.2f} pp")
    print()
    print("  All results saved to:", out_dir)
    print("  Files written:")
    for p in sorted(out_dir.glob("speech_*.jsonl")):
        print(f"    {p}")
    print()
    print("  [STOP — awaiting confirmation before proceeding to RecSys or ResNet-50/DeiT]")
    print("="*70 + "\n")

    _save_results(all_results, out_dir / "speech_all_results.jsonl")


if __name__ == "__main__":
    main()
