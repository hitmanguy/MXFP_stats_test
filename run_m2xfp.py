"""
run_m2xfp.py
============
Dedicated runner for M2XFP format evaluation across all modalities and models.

Reads a single config file (m2xfp_eval.yaml or m2xfp_diagnostic.yaml) that
lists multiple models per modality, then dispatches to the exact same
harnesses used by run_sweep.py — no code duplication.

Usage:
    # Full eval (language + vision + speech) across all configured models:
    python run_m2xfp.py --config configs/m2xfp_eval.yaml

    # Layer-wise NMSE diagnostics across language models:
    python run_m2xfp.py --config configs/m2xfp_diagnostic.yaml --diagnostic

    # Restrict to specific modalities only:
    python run_m2xfp.py --config configs/m2xfp_eval.yaml --only language vision

Results are written to:
    results/m2xfp_eval_<timestamp>.json       (eval mode)
    results/zeroshot/m2xfp_diag_<model>.png   (diagnostic mode)
    and logged to results/eval_ledger.db + results/eval_ledger.jsonl as usual.
"""
from __future__ import annotations

# ── HF cache + SSL fix (must be before any HF/datasets import) ──────────────
import os as _os
from pathlib import Path as _Path

_cache = _Path(_os.environ.get("HF_HOME", str(_Path.home() / "hf_cache")))
_os.environ.setdefault("HF_HOME", str(_cache))
_os.environ.setdefault("HF_DATASETS_CACHE", str(_cache / "datasets"))
_os.environ.setdefault("HF_HUB_CACHE", str(_cache / "hub"))
try:
    import certifi as _certifi
    _os.environ.setdefault("SSL_CERT_FILE", _certifi.where())
    _os.environ.setdefault("REQUESTS_CA_BUNDLE", _certifi.where())
except ImportError:
    pass
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import gc
import json
import traceback
from datetime import datetime
from typing import Any, Dict, List

import torch
import yaml


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_config(path: str) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _get_device(cfg: Dict) -> torch.device:
    d = cfg.get("device", "auto")
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(d)


def _flush_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _merge(base: Dict, override: Dict) -> Dict:
    """Shallow merge: override wins on key collision."""
    out = dict(base)
    out.update(override)
    return out


def _print_banner(text: str):
    bar = "═" * 70
    print(f"\n{bar}")
    print(f"  {text}")
    print(f"{bar}")


# ─────────────────────────────────────────────────────────────────────────────
# Language evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_language(global_cfg: Dict, modality_cfg: Dict, logger, results_dir: str) -> List[Dict]:
    from frameworks.language import LanguageEvalHarness
    from core.quantizer import bits_per_value
    from frameworks.language import _resolve_modes

    device      = _get_device(global_cfg)
    block_size  = global_cfg.get("block_size", 32)
    seeds       = modality_cfg.get("seeds", [42])
    n_chunks    = modality_cfg.get("n_chunks", 50)
    seq_len     = modality_cfg.get("seq_len", 1024)
    quant_modes = modality_cfg.get("quant_modes", ["m2xfp"])
    models      = modality_cfg.get("models", [])

    all_results: List[Dict] = []

    for model_spec in models:
        model_name       = model_spec["model_name"]
        extra_skip_names = model_spec.get("extra_skip_names", [])
        _print_banner(f"LANGUAGE | {model_name}")

        for seed in seeds:
            for mode in quant_modes:
                _flush_gpu()
                print(f"\n  [RUN] mode={mode} | seed={seed}")
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

                    eff_bits = bits_per_value(mode)
                    weight_mode, act_mode = _resolve_modes(mode)

                    logger.log(
                        model_family=model_name,
                        modality="language",
                        dataset="wikitext-2-raw-v1",
                        seed=seed,
                        quant_mode=mode,
                        metric_name="ppl",
                        metric_value=result["ppl"],
                        weight_bits=bits_per_value(weight_mode),
                        act_bits=bits_per_value(act_mode),
                        eff_bits=eff_bits,
                    )
                    all_results.append(result)
                    print(f"  ✓ PPL = {result['ppl']:.4f}  (eff_bits={eff_bits:.2f})")

                except Exception as e:
                    print(f"  [ERROR] {e}")
                    traceback.print_exc()

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Vision evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_vision(global_cfg: Dict, modality_cfg: Dict, logger, results_dir: str) -> List[Dict]:
    from frameworks.vision import VisionEvalHarness
    from core.quantizer import bits_per_value
    from frameworks.language import _resolve_modes

    device      = _get_device(global_cfg)
    block_size  = global_cfg.get("block_size", 32)
    seeds       = modality_cfg.get("seeds", [42])
    n_batches   = modality_cfg.get("n_batches", 100)
    batch_size  = modality_cfg.get("batch_size", 32)
    quant_modes = modality_cfg.get("quant_modes", ["m2xfp"])
    models      = modality_cfg.get("models", [])

    all_results: List[Dict] = []

    for model_spec in models:
        model_name = model_spec["model_name"]
        _print_banner(f"VISION | {model_name}")

        for seed in seeds:
            for mode in quant_modes:
                _flush_gpu()
                print(f"\n  [RUN] mode={mode} | seed={seed}")
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

                    eff_bits = bits_per_value(mode)
                    weight_mode, act_mode = _resolve_modes(mode)
                    w_bits = bits_per_value(weight_mode)
                    a_bits = bits_per_value(act_mode)

                    for metric in ("acc1", "acc5"):
                        logger.log(
                            model_family=model_name,
                            modality="vision",
                            dataset="imagenet-1k",
                            seed=seed,
                            quant_mode=mode,
                            metric_name=metric,
                            metric_value=result[metric],
                            weight_bits=w_bits,
                            act_bits=a_bits,
                            eff_bits=eff_bits,
                        )
                    all_results.append(result)
                    print(f"  ✓ Acc@1={result['acc1']:.2f}%  Acc@5={result['acc5']:.2f}%")

                except Exception as e:
                    print(f"  [ERROR] {e}")
                    traceback.print_exc()

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Speech evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_speech(global_cfg: Dict, modality_cfg: Dict, logger, results_dir: str) -> List[Dict]:
    from frameworks.speech import SpeechEvalHarness
    from core.quantizer import bits_per_value
    from frameworks.language import _resolve_modes

    device      = _get_device(global_cfg)
    block_size  = global_cfg.get("block_size", 32)
    seeds       = modality_cfg.get("seeds", [42])
    n_samples   = modality_cfg.get("n_samples", 200)
    quant_modes = modality_cfg.get("quant_modes", ["m2xfp"])
    # extra_skip_names can be set at modality level (shared across models)
    modal_skip  = modality_cfg.get("extra_skip_names", [])
    models      = modality_cfg.get("models", [])

    all_results: List[Dict] = []

    for model_spec in models:
        model_name  = model_spec["model_name"]
        model_skip  = model_spec.get("extra_skip_names", [])
        # Model-level skip overrides modality-level if present, else inherit
        skip_names  = model_skip if model_skip else modal_skip
        _print_banner(f"SPEECH | {model_name}")

        for seed in seeds:
            for mode in quant_modes:
                _flush_gpu()
                print(f"\n  [RUN] mode={mode} | seed={seed}")
                try:
                    harness = SpeechEvalHarness(
                        model_name=model_name,
                        quant_mode=mode,
                        n_samples=n_samples,
                        seed=seed,
                        block_size=block_size,
                        device=device,
                        extra_skip_names=skip_names,
                    )
                    result = harness.run()
                    result["seed"] = seed

                    eff_bits = bits_per_value(mode)
                    weight_mode, act_mode = _resolve_modes(mode)

                    logger.log(
                        model_family=model_name,
                        modality="speech",
                        dataset="librispeech-test-clean",
                        seed=seed,
                        quant_mode=mode,
                        metric_name="wer_pct",
                        metric_value=result["wer_pct"],
                        weight_bits=bits_per_value(weight_mode),
                        act_bits=bits_per_value(act_mode),
                        eff_bits=eff_bits,
                    )
                    all_results.append(result)
                    print(f"  ✓ WER = {result['wer_pct']:.2f}%")

                except Exception as e:
                    print(f"  [ERROR] {e}")
                    traceback.print_exc()

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic (layer-wise NMSE)
# ─────────────────────────────────────────────────────────────────────────────

def run_diagnostic(global_cfg: Dict, results_dir: str):
    """
    Run the existing run_layer_diagnostics() function from run_diagnostics.py
    for each model listed under diagnostic_models in the config.
    Outputs per-model PNG charts to results/zeroshot/ (reuses existing pipeline).
    """
    from run_diagnostics import run_layer_diagnostics

    block_size   = global_cfg.get("block_size", 32)
    calib_chunks = global_cfg.get("calib_chunks", 64)
    quant_modes  = global_cfg.get("quant_modes", ["m2xfp"])
    device       = global_cfg.get("device", "auto")
    models       = global_cfg.get("diagnostic_models", [])

    for model_spec in models:
        model_name  = model_spec["model_name"]
        seq_len     = model_spec.get("seq_len", 1024)
        extra_skip  = model_spec.get("extra_skip_names", [])

        _print_banner(f"DIAGNOSTIC | {model_name}")

        # Build a per-model cfg dict that matches what run_layer_diagnostics expects
        model_cfg = {
            "model_name":        model_name,
            "seq_len":           seq_len,
            "block_size":        block_size,
            "device":            device,
            "quant_modes":       quant_modes,
            "calib_chunks":      calib_chunks,
            "extra_skip_names":  extra_skip,
        }

        try:
            run_layer_diagnostics(model_cfg)
        except Exception as e:
            print(f"  [ERROR] Diagnostic failed for {model_name}: {e}")
            traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

MODALITY_RUNNERS = {
    "language": run_language,
    "vision":   run_vision,
    "speech":   run_speech,
}


def main():
    parser = argparse.ArgumentParser(
        description="M2XFP dedicated eval runner — multi-modality, multi-model"
    )
    parser.add_argument("--config",     required=True, help="Path to YAML config")
    parser.add_argument("--diagnostic", action="store_true",
                        help="Run layer-wise NMSE diagnostics instead of full eval")
    parser.add_argument("--only",       nargs="+",
                        metavar="MODALITY",
                        help="Only run specified modalities (e.g. --only language vision)")
    parser.add_argument("--results-dir", default="results",
                        help="Directory for SQLite/JSONL logs (default: results/)")
    args = parser.parse_args()

    cfg = _load_config(args.config)

    if args.diagnostic:
        run_diagnostic(cfg, args.results_dir)
        print("\n[run_m2xfp] Diagnostic complete.")
        return

    # ── Eval mode ─────────────────────────────────────────────────────────────
    from core.metrics import EvalLogger
    logger = EvalLogger(results_dir=args.results_dir)

    # Which modalities to run
    configured = cfg.get("modalities")
    if not configured:
        configured = [m for m in MODALITY_RUNNERS.keys() if m in cfg]
        
    if args.only:
        configured = [m for m in configured if m in args.only]

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results: Dict[str, Any] = {}

    for modality in configured:
        if modality not in cfg:
            print(f"[WARN] Modality '{modality}' listed in modalities but has no config block. Skipping.")
            continue
        if modality not in MODALITY_RUNNERS:
            print(f"[WARN] Modality '{modality}' has no runner implemented. Skipping.")
            continue

        modality_cfg = cfg[modality]
        runner       = MODALITY_RUNNERS[modality]
        results      = runner(cfg, modality_cfg, logger, args.results_dir)
        all_results[modality] = results

    # ── Save combined JSON ─────────────────────────────────────────────────────
    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(args.results_dir, f"m2xfp_eval_{timestamp}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n[run_m2xfp] All modalities complete.")
    print(f"  Combined results → {out_path}")
    print(f"  DB log          → {args.results_dir}/eval_ledger.db")


if __name__ == "__main__":
    import os
    main()
