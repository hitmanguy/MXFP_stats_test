import os
import json
import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any

# Ensure matplotlib works in headless environments
import matplotlib
matplotlib.use('Agg')

def load_config(path: str) -> Dict:
    import yaml
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def hook_fn(module, input, output, layer_name, cache):
    """Saves the layer's original inputs and outputs."""
    cache[layer_name] = {
        'input': input[0].detach().cpu(),
        'output': output.detach().cpu()
    }

def get_linear_layers(model, prefix=''):
    """Extract all linear/conv1d layers that we quantize."""
    from transformers.pytorch_utils import Conv1D as HF_Conv1D
    layers = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, HF_Conv1D)):
            layers[name] = module
    return layers

def run_layer_diagnostics(cfg: Dict):
    from frameworks.language import LanguageEvalHarness
    from core.layers import _quantise_weight, _quantise_activation
    import math

    model_name = cfg.get("model_name", "meta-llama/Llama-2-7b-hf")
    seq_len = cfg.get("seq_len", 1024)
    block_size = cfg.get("block_size", 32)
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    modes = cfg.get("quant_modes", ["mxfp4", "mxfp4_residual", "mxfp4_residual_weight_only", "mxfp4_residual_act_only"])
    n_chunks = cfg.get("calib_chunks", 32)  # Default to 32 chunks for stable statistics
    
    print(f"Loading {model_name} in bf16...")
    # Load original unquantized model
    harness = LanguageEvalHarness(model_name=model_name, quant_mode="bf16", seq_len=seq_len, device=device)
    model, tokenizer = harness._load_model_and_tokenizer()
    model.eval()
    
    # Load chunks for calibration
    from frameworks.language import _load_wikitext_chunks
    print(f"Loading {n_chunks} calibration chunks...")
    chunks = _load_wikitext_chunks(tokenizer, seq_len=seq_len, n_chunks=n_chunks, seed=42)
    
    # Move model to device
    model.to(device)

    layers = get_linear_layers(model)
    target_layer_keys = list(layers.keys())
    print(f"Found {len(target_layer_keys)} linear layers. Targeting all layers for full-depth analysis.")

    # We will accumulate signal power and noise power across all chunks
    # signal_power: sum(y_base ** 2)
    # noise_power: sum((y_base - y_q) ** 2)
    
    # Initialize accumulators
    accumulators = {mode: {name: {"signal_power": 0.0, "noise_power": 0.0, "count": 0} for name in target_layer_keys} for mode in modes}

    print(f"Processing {n_chunks} chunks to compute stable reconstruction error...")
    
    # We will process one chunk at a time to avoid OOM, but we need to compute intermediate outputs.
    # To do this efficiently, we can use a forward hook that directly computes the error, 
    # instead of caching all inputs and outputs for all layers for all chunks.
    
    def eval_hook_fn(module, input, output, layer_name, accum):
        # input[0] is the activation, output is y_base
        with torch.no_grad():
            orig_weight = module.weight.data
            x_in = input[0].to(torch.bfloat16)
            y_base = output.to(torch.float32)
            
            sig_pow = torch.sum(y_base ** 2).item()
            num_elements = y_base.numel()
            
            for mode in modes:
                if mode == "fp32" or mode == "bf16":
                    accum[mode][layer_name]["signal_power"] += sig_pow
                    accum[mode][layer_name]["noise_power"] += 0.0
                    accum[mode][layer_name]["count"] += num_elements
                    continue
                    
                from frameworks.language import _resolve_modes
                w_mode, a_mode = _resolve_modes(mode)
                
                import torch.nn.functional as F
                if isinstance(module, nn.Linear):
                    w_q, _ = _quantise_weight(orig_weight, w_mode, block_size)
                    w_q = w_q.to(orig_weight.dtype)
                    
                    x_q, _ = _quantise_activation(x_in.float(), a_mode, block_size)
                    x_q = x_q.to(x_in.dtype)
                    
                    bias = module.bias.to(x_q.dtype) if module.bias is not None else None
                    y_q = F.linear(x_q, w_q, bias)
                else: # Conv1D
                    w_t = orig_weight.t().contiguous()
                    w_q, _ = _quantise_weight(w_t, w_mode, block_size)
                    w_q = w_q.t().contiguous().to(orig_weight.dtype)
                    
                    x_q, _ = _quantise_activation(x_in.float(), a_mode, block_size)
                    x_q = x_q.to(x_in.dtype)
                    
                    bias = module.bias.to(x_q.dtype) if module.bias is not None else None
                    size_out = x_q.shape[:-1] + (w_q.shape[1],)
                    if bias is not None:
                        y_q = torch.addmm(bias, x_q.view(-1, x_q.shape[-1]), w_q)
                    else:
                        y_q = x_q.view(-1, x_q.shape[-1]) @ w_q
                    y_q = y_q.view(size_out)
                    
                noise_pow = torch.sum((y_base - y_q.float()) ** 2).item()
                
                accum[mode][layer_name]["signal_power"] += sig_pow
                accum[mode][layer_name]["noise_power"] += noise_pow
                accum[mode][layer_name]["count"] += num_elements

    # Register hooks
    handles = []
    for name in target_layer_keys:
        h = layers[name].register_forward_hook(
            lambda m, i, o, n=name: eval_hook_fn(m, i, o, n, accumulators)
        )
        handles.append(h)
        
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            for i in range(n_chunks):
                calib_input_ids = chunks[i].unsqueeze(0).to(device)
                model(calib_input_ids)
                if (i+1) % 4 == 0:
                    print(f"  Processed {i+1}/{n_chunks} chunks...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
    for h in handles:
        h.remove()

    # Compute final metrics per layer for each mode
    print("Computing SQNR and NMSE metrics...")
    results_sqnr = {mode: [] for mode in modes}
    results_nmse = {mode: [] for mode in modes}
    
    for mode in modes:
        for name in target_layer_keys:
            sig_pow = accumulators[mode][name]["signal_power"]
            noise_pow = accumulators[mode][name]["noise_power"]
            
            if noise_pow == 0:
                sqnr = 100.0  # Cap infinity at 100 dB for plotting
                nmse = 0.0
            else:
                sqnr = 10 * math.log10(sig_pow / noise_pow)
                nmse = noise_pow / sig_pow
                
            results_sqnr[mode].append(sqnr)
            results_nmse[mode].append(nmse)

    # 3. Plotting
    print("Generating chart...")
    os.makedirs("results", exist_ok=True)
    
    x = np.arange(len(target_layer_keys))
    width = 0.8 / len(modes)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(14, len(target_layer_keys)*0.25), 10))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, mode in enumerate(modes):
        offset = (i - len(modes)/2 + 0.5) * width
        # Plot SQNR (Higher is better)
        ax1.bar(x + offset, results_sqnr[mode], width, label=mode, color=colors[i % len(colors)])
        # Plot NMSE (Lower is better)
        ax2.bar(x + offset, results_nmse[mode], width, label=mode, color=colors[i % len(colors)])
        
    ax1.set_ylabel('SQNR (dB) ↑')
    ax1.set_title(f'Layer-wise Signal-to-Quantization-Noise Ratio ({model_name}, {n_chunks} chunks)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([]) # Hide x labels for top plot
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    ax2.set_ylabel('Normalized MSE (NMSE) ↓')
    ax2.set_title(f'Layer-wise Normalized Mean Squared Error ({model_name}, {n_chunks} chunks)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([k.split('.')[-1] + f"\n({k.split('.')[2]})" if len(k.split('.')) > 2 else k for k in target_layer_keys], rotation=90, ha='center', fontsize=8)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    safe_name = model_name.replace('/', '_').replace('-', '_')
    chart_path = f"results/layer_diagnostics_{safe_name}.png"
    plt.savefig(chart_path, dpi=300)
    print(f"Chart saved to {chart_path}")
    
    # Save JSON data
    data_path = f"results/layer_diagnostics_{safe_name}.json"
    json_data = {
        "model": model_name,
        "calibration_chunks": n_chunks,
        "layers": target_layer_keys,
        "metrics": {
            "SQNR_dB": results_sqnr,
            "NMSE": results_nmse,
        },
        "raw_accumulators": accumulators
    }
    with open(data_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Data saved to {data_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    run_layer_diagnostics(cfg)
