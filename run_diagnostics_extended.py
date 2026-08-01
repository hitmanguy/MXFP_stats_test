"""
run_diagnostics_extended.py
===========================
Extended per-layer reconstruction-error diagnostic.

Builds on the existing run_diagnostics.py but adds support for:
  - wav2vec2 (nn.Conv1d feature extractor + nn.Linear transformer encoder)
  - ResNet-18 / ResNet-50 (nn.Conv2d throughout)

For each model × mode the script accumulates:
  signal_power = Σ y_base²
  noise_power  = Σ (y_base - y_q)²
across all calibration samples, then reports:
  NMSE = noise_power / signal_power   (dimensionless, lower is better)
  SQNR = 10 log10(signal_power / noise_power)  dB

Output (all written to results/):
  layer_diagnostics_<model>_depth.png        — per-layer NMSE vs depth
  layer_diagnostics_<model>_macro_bars.png   — aggregated by section
  layer_diagnostics_<model>.json             — raw numbers
  diagnostic_summary.md                     — 3-5 sentence findings

Usage:
  python run_diagnostics_extended.py --config configs/diagnostic_vision_speech.yaml
  python run_diagnostics_extended.py --model wav2vec2   # inline shortcut
  python run_diagnostics_extended.py --model resnet18
  python run_diagnostics_extended.py --model resnet50
"""
from __future__ import annotations

import os
import json
import math
import argparse
import re
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# Config helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> Dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Generic hook-based accumulator (works for Linear, Conv1d, Conv2d)
# ──────────────────────────────────────────────────────────────────────────────

def _make_eval_hook(layer_name: str, modes: List[str], block_size: int, accumulators: Dict):
    """
    Returns a forward hook that, for each quantization mode, computes
    fake-quantized output from the *same* (weight, activation) pair and
    accumulates (signal_power, noise_power, count).

    Works for nn.Linear, nn.Conv1d, and nn.Conv2d.
    """
    from core.layers import _quantise_weight, _quantise_activation
    from frameworks.language import _resolve_modes

    def hook(module, input, output):
        with torch.no_grad():
            orig_weight = module.weight.data
            x_in  = input[0].float()
            y_base = output.float()

            sig_pow     = torch.sum(y_base ** 2).item()
            num_elements = y_base.numel()

            for mode in modes:
                acc = accumulators[mode][layer_name]
                if mode in ("fp32", "bf16"):
                    acc["signal_power"] += sig_pow
                    acc["noise_power"]  += 0.0
                    acc["count"]        += num_elements
                    continue

                w_mode, a_mode = _resolve_modes(mode)

                # ── Weight quantization ────────────────────────────────────
                if isinstance(module, nn.Conv1d):
                    # Conv1d weight: [out, in/groups, kW] — reshape to 2D for blocker
                    w_2d = orig_weight.reshape(orig_weight.shape[0], -1)
                    w_q_2d, _ = _quantise_weight(w_2d, w_mode, block_size)
                    w_q = w_q_2d.reshape(orig_weight.shape).to(orig_weight.dtype)
                elif isinstance(module, nn.Conv2d):
                    # Conv2d weight: [out, in/groups, kH, kW]
                    w_2d = orig_weight.reshape(orig_weight.shape[0], -1)
                    w_q_2d, _ = _quantise_weight(w_2d, w_mode, block_size)
                    w_q = w_q_2d.reshape(orig_weight.shape).to(orig_weight.dtype)
                else:
                    # nn.Linear
                    w_q, _ = _quantise_weight(orig_weight, w_mode, block_size)
                    w_q = w_q.to(orig_weight.dtype)

                # ── Activation quantization ────────────────────────────────
                # _quantise_activation expects a 2D tensor (or flattens internally);
                # flatten spatial/sequence dims, quantize, restore.
                orig_shape = x_in.shape
                x_flat = x_in.reshape(-1, orig_weight.shape[1] if isinstance(module, (nn.Conv2d, nn.Conv1d)) else orig_weight.shape[1])
                # Use the flat view for quantization, then reshape back
                x_q_flat, _ = _quantise_activation(x_flat, a_mode, block_size)
                x_q = x_q_flat.reshape(orig_shape).to(input[0].dtype)

                # ── Re-run layer op ────────────────────────────────────────
                bias = module.bias.to(x_q.dtype) if module.bias is not None else None
                if isinstance(module, nn.Linear):
                    y_q = F.linear(x_q, w_q, bias)
                elif isinstance(module, nn.Conv1d):
                    y_q = F.conv1d(x_q, w_q, bias,
                                   module.stride, module.padding,
                                   module.dilation, module.groups)
                else:  # Conv2d
                    y_q = F.conv2d(x_q, w_q, bias,
                                   module.stride, module.padding,
                                   module.dilation, module.groups)

                noise_pow = torch.sum((y_base - y_q.float()) ** 2).item()

                acc["signal_power"] += sig_pow
                acc["noise_power"]  += noise_pow
                acc["count"]        += num_elements

                del w_q, x_q, y_q

    return hook


# ──────────────────────────────────────────────────────────────────────────────
# Layer discovery
# ──────────────────────────────────────────────────────────────────────────────

def get_target_layers(
    model: nn.Module,
    include_types: Tuple = (nn.Linear, nn.Conv1d, nn.Conv2d),
    skip_names: List[str] = None,
) -> Dict[str, nn.Module]:
    """
    Walk model and return {full_name: module} for all matching layer types,
    excluding any name in skip_names.
    """
    if skip_names is None:
        skip_names = []
    skip_set = set(skip_names)
    layers = {}
    for name, module in model.named_modules():
        if isinstance(module, include_types) and name not in skip_set:
            layers[name] = module
    return layers


# ──────────────────────────────────────────────────────────────────────────────
# NMSE computation from accumulators
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(accumulators: Dict, modes: List[str], layer_keys: List[str]):
    results_sqnr = {m: [] for m in modes}
    results_nmse = {m: [] for m in modes}

    for mode in modes:
        for name in layer_keys:
            sig_pow   = accumulators[mode][name]["signal_power"]
            noise_pow = accumulators[mode][name]["noise_power"]

            if noise_pow == 0 or sig_pow == 0:
                sqnr = 100.0
                nmse = 0.0
            else:
                sqnr = 10 * math.log10(max(sig_pow / noise_pow, 1e-12))
                nmse = noise_pow / sig_pow

            results_sqnr[mode].append(sqnr)
            results_nmse[mode].append(nmse)

    return results_sqnr, results_nmse


# ──────────────────────────────────────────────────────────────────────────────
# Localization metric
# ──────────────────────────────────────────────────────────────────────────────

def localization_score(nmse_list: List[float], top_fraction: float = 0.10) -> float:
    """
    Fraction of total NMSE attributable to the top (highest-error) layer fraction.
    Matches the definition used in the existing motivation writeup.
    """
    arr = np.array(nmse_list)
    total = arr.sum()
    if total == 0:
        return 0.0
    k = max(1, int(np.ceil(len(arr) * top_fraction)))
    top_k_sum = np.partition(arr, -k)[-k:].sum()
    return top_k_sum / total


# ──────────────────────────────────────────────────────────────────────────────
# Plotting  (same style as run_diagnostics.py)
# ──────────────────────────────────────────────────────────────────────────────

def plot_depth_chart(
    layer_keys: List[str],
    results_nmse: Dict[str, List[float]],
    modes: List[str],
    model_name: str,
    out_dir: str,
    section_labels: Dict[str, str] = None,   # name → section tag for bar chart
):
    """Replicates the 'faceted depth' and 'macro bars' style from run_diagnostics.py."""
    import pandas as pd
    import seaborn as sns

    safe = model_name.replace("/", "_").replace("-", "_")
    os.makedirs(out_dir, exist_ok=True)

    # ── Build data frame ──────────────────────────────────────────────────────
    plot_modes = [m for m in modes if m not in ("fp32", "bf16")]
    data = []
    for mode in plot_modes:
        for i, name in enumerate(layer_keys):
            section = (section_labels or {}).get(name, "other")
            data.append({
                "layer_name":  name,
                "layer_index": i,
                "section":     section,
                "format":      mode,
                "nmse":        results_nmse[mode][i],
            })

    if not data:
        return
    df = pd.DataFrame(data)
    sns.set_theme(style="whitegrid")

    # 1. Full-depth line chart
    fig, ax = plt.subplots(figsize=(max(12, len(layer_keys) // 4), 5))
    for mode in plot_modes:
        sub = df[df["format"] == mode]
        ax.plot(sub["layer_index"], sub["nmse"], marker="o", markersize=3,
                linewidth=1.4, label=mode)
    ax.set_xlabel("Layer depth (index)")
    ax.set_ylabel("NMSE (lower is better)")
    ax.set_title(f"Per-layer NMSE by depth — {model_name}")
    ax.legend(title="Format")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    fig.tight_layout()
    depth_path = os.path.join(out_dir, f"layer_diagnostics_{safe}_depth.png")
    fig.savefig(depth_path, dpi=300)
    plt.close(fig)
    print(f"  Depth chart → {depth_path}")

    # 2. Section-aggregated bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    section_df = df.groupby(["section", "format"])["nmse"].mean().reset_index()
    sections   = section_df["section"].unique()
    x = np.arange(len(sections))
    width = 0.8 / max(len(plot_modes), 1)
    for j, mode in enumerate(plot_modes):
        vals = [section_df[(section_df["section"] == s) & (section_df["format"] == mode)]["nmse"].values
                for s in sections]
        vals = [v[0] if len(v) > 0 else 0.0 for v in vals]
        ax.bar(x + j * width - (len(plot_modes) - 1) * width / 2, vals,
               width=width, label=mode)
    ax.set_xticks(x)
    ax.set_xticklabels(sections, rotation=30, ha="right")
    ax.set_ylabel("Mean NMSE")
    ax.set_title(f"NMSE aggregated by section — {model_name}")
    ax.legend(title="Format")
    fig.tight_layout()
    macro_path = os.path.join(out_dir, f"layer_diagnostics_{safe}_macro_bars.png")
    fig.savefig(macro_path, dpi=300)
    plt.close(fig)
    print(f"  Macro bars → {macro_path}")

    return depth_path, macro_path


# ──────────────────────────────────────────────────────────────────────────────
# WAV2VEC2 diagnostic
# ──────────────────────────────────────────────────────────────────────────────

def run_wav2vec2_diagnostic(cfg: Dict, out_dir: str):
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    from datasets import load_dataset

    model_name = cfg.get("model_name", "facebook/wav2vec2-base-960h")
    block_size  = cfg.get("block_size", 32)
    n_samples   = cfg.get("n_samples", 50)
    modes       = cfg.get("quant_modes", ["mxfp4", "mxfp4_residual_act_only", "mxfp4_residual_weight_only"])
    device      = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    skip_names  = cfg.get("extra_skip_names", ["lm_head"])

    print(f"\n{'='*60}")
    print(f"  DIAGNOSTIC: {model_name}")
    print(f"  Modes: {modes}  |  n_samples: {n_samples}")
    print(f"{'='*60}\n")

    print(f"  Loading {model_name}...")
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model     = Wav2Vec2ForCTC.from_pretrained(model_name, torch_dtype=torch.float32)
    model.eval().to(device)

    # ── Layer discovery ───────────────────────────────────────────────────────
    layers = get_target_layers(
        model,
        include_types=(nn.Linear, nn.Conv1d),
        skip_names=skip_names,
    )
    layer_keys = list(layers.keys())
    print(f"  Found {len(layer_keys)} layers (Conv1d + Linear, excl. {skip_names})")

    # Tag each layer's section for the bar chart
    def _section(name: str) -> str:
        if "feature_extractor" in name or "feature_projection" in name:
            return "conv_frontend"
        return "transformer"

    section_labels = {name: _section(name) for name in layer_keys}

    # ── Accumulators ──────────────────────────────────────────────────────────
    accumulators = {
        mode: {name: {"signal_power": 0.0, "noise_power": 0.0, "count": 0}
               for name in layer_keys}
        for mode in modes
    }

    # ── Load calibration samples (LibriSpeech test-clean) ────────────────────
    print(f"  Loading {n_samples} LibriSpeech samples...")
    ds = load_dataset("openslr/librispeech_asr", "clean", split="test",
                      streaming=True, trust_remote_code=True)
    samples = []
    for item in ds:
        samples.append(item["audio"]["array"])
        if len(samples) >= n_samples:
            break
    print(f"  Loaded {len(samples)} samples.")

    # ── Register hooks ────────────────────────────────────────────────────────
    handles = []
    for name in layer_keys:
        h = layers[name].register_forward_hook(
            _make_eval_hook(name, modes, block_size, accumulators)
        )
        handles.append(h)

    # ── Forward pass over calibration data ───────────────────────────────────
    print(f"  Running forward passes...")
    with torch.no_grad():
        for i, audio in enumerate(samples):
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt",
                               padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            model(**inputs)
            if (i + 1) % 10 == 0:
                print(f"    {i+1}/{n_samples} samples processed...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    for h in handles:
        h.remove()

    # ── Metrics ───────────────────────────────────────────────────────────────
    results_sqnr, results_nmse = compute_metrics(accumulators, modes, layer_keys)

    # ── Localization metric ───────────────────────────────────────────────────
    reference_mode = "mxfp4"
    loc_score = localization_score(results_nmse[reference_mode]) if reference_mode in modes else None

    # ── Section-split NMSE (Conv frontend vs Transformer) ─────────────────────
    section_summary = {}
    for mode in modes:
        conv_nmse = [results_nmse[mode][i] for i, k in enumerate(layer_keys)
                     if section_labels[k] == "conv_frontend"]
        xfmr_nmse = [results_nmse[mode][i] for i, k in enumerate(layer_keys)
                     if section_labels[k] == "transformer"]
        section_summary[mode] = {
            "conv_frontend_mean_nmse": float(np.mean(conv_nmse)) if conv_nmse else None,
            "transformer_mean_nmse":   float(np.mean(xfmr_nmse)) if xfmr_nmse else None,
        }
    print("\n  Section NMSE summary:")
    for mode, s in section_summary.items():
        print(f"    [{mode}]  conv_frontend={s['conv_frontend_mean_nmse']:.6f}  "
              f"transformer={s['transformer_mean_nmse']:.6f}")

    # ── Plotting ──────────────────────────────────────────────────────────────
    plot_depth_chart(layer_keys, results_nmse, modes, model_name, out_dir, section_labels)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    safe = model_name.replace("/", "_").replace("-", "_")
    json_path = os.path.join(out_dir, f"layer_diagnostics_{safe}.json")
    with open(json_path, "w") as f:
        json.dump({
            "model": model_name,
            "n_samples": n_samples,
            "layers": layer_keys,
            "section_labels": section_labels,
            "section_summary": section_summary,
            "localization_top10pct_mxfp4": loc_score,
            "metrics": {"SQNR_dB": results_sqnr, "NMSE": results_nmse},
        }, f, indent=2)
    print(f"  Data → {json_path}")

    return results_nmse, layer_keys, section_labels, section_summary, loc_score


# ──────────────────────────────────────────────────────────────────────────────
# VISION (ResNet-18 / ResNet-50) diagnostic
# ──────────────────────────────────────────────────────────────────────────────

def run_vision_diagnostic(cfg: Dict, out_dir: str):
    from datasets import load_dataset as hf_load_dataset
    from PIL import Image
    import torchvision.transforms as T

    model_name = cfg.get("model_name", "resnet18")
    block_size  = cfg.get("block_size", 32)
    n_batches   = cfg.get("n_batches", 20)
    batch_size  = cfg.get("batch_size", 16)
    modes       = cfg.get("quant_modes", ["mxfp4", "mxfp4_residual_act_only", "mxfp4_residual_weight_only"])
    device      = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    print(f"\n{'='*60}")
    print(f"  DIAGNOSTIC: {model_name}")
    print(f"  Modes: {modes}  |  n_batches×batch_size: {n_batches}×{batch_size}")
    print(f"{'='*60}\n")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"  Loading {model_name}...")
    from torchvision.models import (
        resnet18, ResNet18_Weights,
        resnet50, ResNet50_Weights,
    )
    if model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT
        model   = resnet18(weights=weights)
        transform = weights.transforms()
    elif model_name == "resnet50":
        weights = ResNet50_Weights.DEFAULT
        model   = resnet50(weights=weights)
        transform = weights.transforms()
    else:
        raise ValueError(f"Unsupported vision model: {model_name}")

    model.eval().to(device)

    # ── Layer discovery: all Conv2d layers ────────────────────────────────────
    layers = get_target_layers(model, include_types=(nn.Conv2d,))
    layer_keys = list(layers.keys())
    print(f"  Found {len(layer_keys)} Conv2d layers")

    # Simple section tag: name the residual stage from the name
    def _stage(name: str) -> str:
        m = re.match(r"layer(\d)", name)
        return f"stage{m.group(1)}" if m else ("stem" if "conv1" in name else "head")

    section_labels = {name: _stage(name) for name in layer_keys}

    # ── Accumulators ──────────────────────────────────────────────────────────
    accumulators = {
        mode: {name: {"signal_power": 0.0, "noise_power": 0.0, "count": 0}
               for name in layer_keys}
        for mode in modes
    }

    # ── Load calibration images (ImageNet-1k val, streaming) ─────────────────
    print(f"  Loading ImageNet-1k (streaming, {n_batches} batches)...")
    ds = hf_load_dataset("ILSVRC/imagenet-1k", split="validation",
                         streaming=True, trust_remote_code=True, token=True)
    ds_iter  = iter(ds)
    n_images = n_batches * batch_size

    # ── Register hooks ────────────────────────────────────────────────────────
    handles = []
    for name in layer_keys:
        h = layers[name].register_forward_hook(
            _make_eval_hook(name, modes, block_size, accumulators)
        )
        handles.append(h)

    # ── Forward loop ──────────────────────────────────────────────────────────
    print(f"  Running forward passes ({n_images} images)...")
    with torch.no_grad():
        for b in range(n_batches):
            batch_imgs = []
            for _ in range(batch_size):
                try:
                    item = next(ds_iter)
                    img  = item["image"]
                    # Some ImageNet samples are grayscale — convert to RGB
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    batch_imgs.append(transform(img))
                except StopIteration:
                    break

            if not batch_imgs:
                break

            x = torch.stack(batch_imgs).to(device)
            model(x)

            if (b + 1) % 5 == 0:
                print(f"    {b+1}/{n_batches} batches processed...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    for h in handles:
        h.remove()

    # ── Metrics ───────────────────────────────────────────────────────────────
    results_sqnr, results_nmse = compute_metrics(accumulators, modes, layer_keys)

    reference_mode = "mxfp4"
    loc_score = localization_score(results_nmse[reference_mode]) if reference_mode in modes else None
    print(f"\n  Localization (top-10% layers share of NMSE, {reference_mode}): {loc_score:.3f}")

    # ── Plotting ──────────────────────────────────────────────────────────────
    plot_depth_chart(layer_keys, results_nmse, modes, model_name, out_dir, section_labels)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    safe = model_name.replace("/", "_").replace("-", "_")
    json_path = os.path.join(out_dir, f"layer_diagnostics_{safe}.json")
    with open(json_path, "w") as f:
        json.dump({
            "model": model_name,
            "n_batches": n_batches,
            "batch_size": batch_size,
            "layers": layer_keys,
            "section_labels": section_labels,
            "localization_top10pct_mxfp4": loc_score,
            "metrics": {"SQNR_dB": results_sqnr, "NMSE": results_nmse},
        }, f, indent=2)
    print(f"  Data → {json_path}")

    return results_nmse, layer_keys, section_labels, loc_score


# ──────────────────────────────────────────────────────────────────────────────
# Summary markdown
# ──────────────────────────────────────────────────────────────────────────────

def write_summary_md(findings: List[Dict], out_path: str):
    lines = [
        "# Extended Layer Diagnostic — Summary of Findings\n",
        "_Auto-generated by `run_diagnostics_extended.py`._\n",
        "---\n",
    ]
    for f in findings:
        lines.append(f"## {f['model']}\n")
        lines.append(f"- **Modes tested:** {', '.join(f['modes'])}\n")
        if f.get("localization") is not None:
            lines.append(
                f"- **Localization score** (fraction of total NMSE in top-10% of layers, "
                f"mxfp4): **{f['localization']:.3f}** "
                f"({'highly localized' if f['localization'] > 0.5 else 'distributed'})\n"
            )
        if f.get("section_summary"):
            lines.append("- **Section NMSE breakdown (wav2vec2):**\n")
            for mode, s in f["section_summary"].items():
                cf = s.get("conv_frontend_mean_nmse")
                xf = s.get("transformer_mean_nmse")
                if cf is not None and xf is not None:
                    dominant = "conv_frontend" if cf > xf else "transformer"
                    lines.append(
                        f"  - `{mode}`: conv_frontend={cf:.5f}, "
                        f"transformer={xf:.5f} → **{dominant} dominates**\n"
                    )
        lines.append("\n")

    # Paragraph summary
    lines.append("---\n")
    lines.append("## Interpretation note\n")
    lines.append(
        "If `conv_frontend_mean_nmse > transformer_mean_nmse` for "
        "`mxfp4_residual_act_only`, the error blowup in wav2vec2 under "
        "activation-residual quantization is localized in the raw-waveform "
        "Conv1d frontend, supporting the 'high dynamic-range input' hypothesis. "
        "If the split is roughly equal or transformer-dominant, the "
        "cancellation mechanism (a·Δw vs Δa·w) is the more likely explanation "
        "and a sign-correlation experiment is warranted. "
        "If the data are ambiguous (ratio < 2× between sections), "
        "both hypotheses remain viable and both experiments are required before "
        "publication.\n"
    )

    with open(out_path, "w") as fh:
        fh.writelines(lines)
    print(f"\n  Summary → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

MODEL_SHORTCUTS = {
    "wav2vec2":   "facebook/wav2vec2-base-960h",
    "resnet18":   "resnet18",
    "resnet50":   "resnet50",
}

def main():
    parser = argparse.ArgumentParser(
        description="Extended per-layer NMSE diagnostic for vision and speech models"
    )
    parser.add_argument("--config", default=None,
                        help="Path to YAML config. Overrides --model flags.")
    parser.add_argument("--model", choices=list(MODEL_SHORTCUTS.keys()),
                        help="Shortcut to run a single model with default settings.")
    parser.add_argument("--results-dir", default="results",
                        help="Output directory for plots, JSON, and markdown.")
    args = parser.parse_args()

    out_dir = args.results_dir
    os.makedirs(out_dir, exist_ok=True)

    default_modes = ["mxfp4", "mxfp4_residual_act_only", "mxfp4_residual_weight_only"]

    # Build job list
    if args.config:
        cfg = load_config(args.config)
        jobs = cfg.get("diagnostic_models", [cfg])   # support single-model or list
    elif args.model:
        jobs = [{
            "model_name": MODEL_SHORTCUTS[args.model],
            "block_size": 32,
            "n_samples":  50,
            "n_batches":  20,
            "batch_size": 16,
            "quant_modes": default_modes,
            "device": "auto",
        }]
    else:
        parser.error("Provide either --config or --model.")

    findings = []

    for job in jobs:
        model_name = job.get("model_name", "")
        device_str = job.get("device", "auto")
        if device_str == "auto":
            job["device"] = "cuda" if torch.cuda.is_available() else "cpu"

        if "wav2vec2" in model_name.lower():
            nmse, keys, sec_labels, sec_summary, loc = run_wav2vec2_diagnostic(job, out_dir)
            findings.append({
                "model": model_name,
                "modes": job.get("quant_modes", default_modes),
                "localization": loc,
                "section_summary": sec_summary,
            })
        elif "resnet" in model_name.lower():
            nmse, keys, sec_labels, loc = run_vision_diagnostic(job, out_dir)
            findings.append({
                "model": model_name,
                "modes": job.get("quant_modes", default_modes),
                "localization": loc,
                "section_summary": None,
            })
        else:
            print(f"[WARN] Unknown model type '{model_name}' — skipping.")

    # Write summary markdown
    md_path = os.path.join(out_dir, "diagnostic_summary.md")
    write_summary_md(findings, md_path)

    print("\n[run_diagnostics_extended] Done.")


if __name__ == "__main__":
    main()
