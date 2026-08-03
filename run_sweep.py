"""
run_sweep.py
============
Master CLI runner for MXFP4 evaluation sweeps.

Usage:
    # Run acceptance gate (Part 3)
    python run_sweep.py --config configs/acceptance_gate.yaml

    # Run full adaptive sweep
    python run_sweep.py --config configs/adaptive_residual_sweep.yaml

    # Quick smoke test (CPU only, 5 chunks)
    python run_sweep.py --config configs/debug_gpt2.yaml

    # Run with significance testing
    python run_sweep.py --config configs/acceptance_gate.yaml --significance
"""
from __future__ import annotations
from pathlib import Path

# ── SSL cert fix (must be first, before any network-aware library imports) ──
import os as _os
if _os.name == "nt":
    cache = Path("D:/hf_cache")
else:
    cache = Path.home() / "hf_cache"

_os.environ["HF_HOME"] = str(cache)
_os.environ["HF_DATASETS_CACHE"] = str(cache / "datasets")
_os.environ["HF_HUB_CACHE"] = str(cache / "hub")

try:
    import certifi as _certifi
    _os.environ.setdefault("SSL_CERT_FILE", _certifi.where())
    _os.environ.setdefault("REQUESTS_CA_BUNDLE", _certifi.where())
except ImportError:
    pass
# ────────────────────────────────────────────────────────────────────────────

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
import torch
import numpy as np


def load_config(path: str) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_language_eval(
    cfg: Dict,
    logger,
    significance: bool = False,
) -> List[Dict]:
    """Run language evaluations as specified by config. Returns list of result dicts."""
    from frameworks.language import LanguageEvalHarness

    model_name = cfg.get("model_name", "gpt2")
    seeds = cfg.get("seeds", [42, 123, 1337])
    n_chunks = cfg.get("n_chunks", 50)
    seq_len = cfg.get("seq_len", 1024)
    block_size = cfg.get("block_size", 32)
    quant_modes = cfg.get("quant_modes", ["fp32"])
    device_str = cfg.get("device", "auto")

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    all_results: List[Dict] = []
    per_chunk_nlls: Dict[str, List[float]] = {}  # mode → nll list (seed=42)

    for seed in seeds:
        for mode in quant_modes:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            print(f"\n[SWEEP] {model_name} | mode={mode} | seed={seed}")
            try:
                harness = LanguageEvalHarness(
                    model_name=model_name,
                    quant_mode=mode,
                    seed=seed,
                    n_chunks=n_chunks,
                    seq_len=seq_len,
                    block_size=block_size,
                    device=device,
                )
                result = harness.run()
                result["seed"] = seed

                # Log to DB + JSONL
                from core.quantizer import bits_per_value
                from frameworks.language import _resolve_modes
                eff_bits = bits_per_value(mode)
                weight_mode, act_mode = _resolve_modes(mode)
                weight_bits = bits_per_value(weight_mode)
                act_bits = bits_per_value(act_mode)
                
                logger.log(
                    model_family=model_name,
                    modality="language",
                    dataset="wikitext-2-raw-v1",
                    seed=seed,
                    quant_mode=mode,
                    metric_name="ppl",
                    metric_value=result["ppl"],
                    weight_bits=weight_bits,
                    act_bits=act_bits,
                    eff_bits=eff_bits,
                    ppl=result.get("ppl"),
                )

                all_results.append(result)

                # Store per-chunk NLL for significance testing from all seeds
                if mode not in per_chunk_nlls:
                    per_chunk_nlls[mode] = []
                per_chunk_nlls[mode].extend(result["per_chunk_nll"])

            except Exception as e:
                print(f"  [ERROR] {e}")
                import traceback
                traceback.print_exc()

    # ── Significance testing ──────────────────────────────────────────────────
    if significance and len(per_chunk_nlls) >= 2:
        _run_significance_tests(per_chunk_nlls)

    return all_results


def _run_significance_tests(per_chunk_nlls: Dict[str, List[float]]) -> None:
    """Run paired significance tests on all adjacent Pareto pairs."""
    from analyze_results import full_significance_report, print_significance_report
    from core.quantizer import bits_per_value

    # Sort modes by eff_bits
    modes_sorted = sorted(per_chunk_nlls.keys(), key=lambda m: bits_per_value(m))

    print("\n" + "="*60)
    print("  PART 4 — Paired Significance Tests")
    print("="*60)

    for i in range(len(modes_sorted) - 1):
        m_a = modes_sorted[i]
        m_b = modes_sorted[i + 1]
        a = np.array(per_chunk_nlls[m_a])
        b = np.array(per_chunk_nlls[m_b])
        report = full_significance_report(m_a, a, m_b, b)
        print_significance_report(report)


def run_vision_eval(
    cfg: Dict,
    logger,
    significance: bool = False,
) -> List[Dict]:
    """Run vision evaluations as specified by config."""
    from frameworks.vision import VisionEvalHarness

    model_name = cfg.get("model_name", "resnet18")
    seeds = cfg.get("seeds", [42, 123, 1337])
    n_batches = cfg.get("n_batches", 100)
    batch_size = cfg.get("batch_size", 32)
    block_size = cfg.get("block_size", 32)
    quant_modes = cfg.get("quant_modes", ["fp32"])
    device_str = cfg.get("device", "auto")

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    all_results: List[Dict] = []

    for seed in seeds:
        for mode in quant_modes:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            print(f"\n[SWEEP] {model_name} | mode={mode} | seed={seed}")
            try:
                harness = VisionEvalHarness(
                    model_name=model_name,
                    quant_mode=mode,
                    n_batches=n_batches,
                    batch_size=batch_size,
                    seed=seed,
                    block_size=block_size,
                    device=device,
                )
                result = harness.run()
                result["seed"] = seed

                from core.quantizer import bits_per_value
                from frameworks.language import _resolve_modes
                eff_bits = bits_per_value(mode)
                weight_mode, act_mode = _resolve_modes(mode)
                weight_bits = bits_per_value(weight_mode)
                act_bits = bits_per_value(act_mode)
                
                logger.log(
                    model_family=model_name,
                    modality="vision",
                    dataset="imagenet-1k",
                    seed=seed,
                    quant_mode=mode,
                    metric_name="acc1",
                    metric_value=result["acc1"],
                    weight_bits=weight_bits,
                    act_bits=act_bits,
                    eff_bits=eff_bits,
                )
                
                logger.log(
                    model_family=model_name,
                    modality="vision",
                    dataset="imagenet-1k",
                    seed=seed,
                    quant_mode=mode,
                    metric_name="acc5",
                    metric_value=result["acc5"],
                    weight_bits=weight_bits,
                    act_bits=act_bits,
                    eff_bits=eff_bits,
                )

                all_results.append(result)
            except Exception as e:
                print(f"  [ERROR] {e}")
                import traceback
                traceback.print_exc()

    return all_results

def run_speech_eval(
    cfg: Dict,
    logger,
) -> List[Dict]:
    """Run speech (WER) evaluations as specified by config."""
    from frameworks.speech import SpeechEvalHarness

    model_name = cfg.get("model_name", "facebook/wav2vec2-base-960h")
    seeds = cfg.get("seeds", [42, 123, 1337])
    n_samples = cfg.get("n_samples", 200)
    block_size = cfg.get("block_size", 32)
    quant_modes = cfg.get("quant_modes", ["fp32"])
    device_str = cfg.get("device", "auto")
    skip_feature_extractor = cfg.get("skip_feature_extractor", False)
    extra_skip_names = cfg.get("extra_skip_names", [])

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    all_results: List[Dict] = []

    for seed in seeds:
        for mode in quant_modes:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            print(f"\n[SWEEP] {model_name} | mode={mode} | seed={seed}")
            try:
                harness = SpeechEvalHarness(
                    model_name=model_name,
                    quant_mode=mode,
                    n_samples=n_samples,
                    seed=seed,
                    block_size=block_size,
                    device=device,
                    skip_feature_extractor=skip_feature_extractor,
                    extra_skip_names=extra_skip_names,
                )
                result = harness.run()
                result["seed"] = seed

                from core.quantizer import bits_per_value
                from frameworks.language import _resolve_modes
                eff_bits = bits_per_value(mode)
                weight_mode, act_mode = _resolve_modes(mode)
                weight_bits = bits_per_value(weight_mode)
                act_bits = bits_per_value(act_mode)

                logger.log(
                    model_family=model_name,
                    modality="speech",
                    dataset="librispeech-test-clean",
                    seed=seed,
                    quant_mode=mode,
                    metric_name="wer_pct",
                    metric_value=result["wer_pct"],
                    weight_bits=weight_bits,
                    act_bits=act_bits,
                    eff_bits=eff_bits,
                )

                all_results.append(result)
            except Exception as e:
                print(f"  [ERROR] {e}")
                import traceback
                traceback.print_exc()

    return all_results


def run_recsys_eval(
    cfg: Dict,
    logger,
) -> List[Dict]:
    """Run recsys (AUC) evaluations as specified by config."""
    from frameworks.recsys import RecSysEvalHarness
    from core.quantizer import bits_per_value
    from frameworks.language import _resolve_modes

    quant_modes = cfg.get("quant_modes", ["fp32"])
    seeds = cfg.get("seeds", [42])
    n_samples = cfg.get("n_samples", 100_000)
    block_size = cfg.get("block_size", 32)
    device_str = cfg.get("device", "auto")
    quantize_embeddings = cfg.get("quantize_embeddings", False)

    import torch
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    all_results = []
    for seed in seeds:
        for mode in quant_modes:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            try:
                harness = RecSysEvalHarness(
                    quant_mode=mode,
                    n_samples=n_samples,
                    seed=seed,
                    block_size=block_size,
                    device=device,
                    quantize_embeddings=quantize_embeddings,
                )
                result = harness.run()
                result["seed"] = seed

                eff_bits = bits_per_value(mode)
                weight_mode, act_mode = _resolve_modes(mode)
                weight_bits = bits_per_value(weight_mode)
                act_bits = bits_per_value(act_mode)

                logger.log(
                    model_family="dlrm",
                    modality="recsys",
                    dataset="criteo-terabyte",
                    seed=seed,
                    quant_mode=mode,
                    metric_name="auc",
                    metric_value=result["auc"],
                    weight_bits=weight_bits,
                    act_bits=act_bits,
                    eff_bits=eff_bits,
                )

                all_results.append(result)
            except Exception as e:
                print(f"  [ERROR] {e}")
                import traceback
                traceback.print_exc()

    return all_results

def main():
    parser = argparse.ArgumentParser(description="MXFP4 Evaluation Sweep Runner")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--significance", action="store_true",
                        help="Run Part 4 significance tests after eval")
    parser.add_argument("--results-dir", default="results",
                        help="Results directory (default: results/)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"\n[run_sweep] Loaded config: {args.config}")
    print(json.dumps(cfg, indent=2))

    from core.metrics import EvalLogger

    class _WrappedLogger(EvalLogger):
        def log(self, ppl=None, **kwargs):
            kwargs.pop("ppl", None)
            return super().log(**kwargs)

    logger = _WrappedLogger(results_dir=args.results_dir)

    modality = cfg.get("modality", "language")
    if modality == "language":
        results = run_language_eval(cfg, logger, significance=args.significance)
    elif modality == "vision":
        results = run_vision_eval(cfg, logger, significance=args.significance)
    elif modality == "speech":
        results = run_speech_eval(cfg, logger)
    elif modality == "recsys":
        results = run_recsys_eval(cfg, logger)
    else:
        print(f"[WARN] Modality '{modality}' not supported")

    print("\n[run_sweep] Sweep complete.")


if __name__ == "__main__":
    main()
