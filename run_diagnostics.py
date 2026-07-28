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

    model_name = cfg.get("model_name", "meta-llama/Llama-2-7b-hf")
    seq_len = cfg.get("seq_len", 1024)
    block_size = cfg.get("block_size", 32)
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    modes = cfg.get("quant_modes", ["mxfp4", "mxfp4_residual", "mxfp4_residual_weight_only", "mxfp4_residual_act_only"])
    
    print(f"Loading {model_name} in bf16...")
    # Load original unquantized model
    harness = LanguageEvalHarness(model_name=model_name, quant_mode="bf16", seq_len=seq_len, device=device)
    model, tokenizer = harness._load_model_and_tokenizer()
    model.eval()
    
    # Load 1 chunk for calibration
    from frameworks.language import _load_wikitext_chunks
    print("Loading calibration chunk...")
    chunks = _load_wikitext_chunks(tokenizer, seq_len=seq_len, n_chunks=1, seed=42)
    calib_input_ids = chunks[0].unsqueeze(0).to(device)

    # Move model to device
    model.to(device)

    layers = get_linear_layers(model)
    print(f"Found {len(layers)} linear layers. Selecting a subset for visualization...")
    
    # Select a subset of layers to visualize (e.g., first, middle, last layer of attention and MLP)
    # We will pick specific layers from the model to avoid plotting 200+ layers which makes a messy chart.
    target_layer_keys = []
    layer_names = list(layers.keys())
    
    # Try to intelligently pick a spread of layers for LLaMA
    for l_idx in [0, len(layer_names)//4, len(layer_names)//2, (3*len(layer_names))//4, len(layer_names)-1]:
        if l_idx < len(layer_names):
            target_layer_keys.append(layer_names[l_idx])
            
    # If the user specifically wants all, we can do it, but 10-15 is best for a chart.
    # Let's just take the first 15 layers of the model for a clean chart
    target_layer_keys = layer_names[:15]
    print(f"Targeting layers: {target_layer_keys}")

    # 1. Capture FP32/BF16 Base Inputs & Outputs
    print("Running base FP32/BF16 forward pass to capture activations...")
    cache = {}
    handles = []
    for name in target_layer_keys:
        h = layers[name].register_forward_hook(
            lambda m, i, o, n=name: hook_fn(m, i, o, n, cache)
        )
        handles.append(h)
        
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            model(calib_input_ids)
            
    for h in handles:
        h.remove()

    # 2. Compute Reconstruction Error per Layer for each mode
    from frameworks.language import _resolve_modes
    results_mse = {mode: [] for mode in modes}
    
    print("Computing reconstruction error per layer...")
    
    with torch.no_grad():
        for name in target_layer_keys:
            layer = layers[name]
            orig_weight = layer.weight.data.clone()
            
            x_in = cache[name]['input'].to(device).to(torch.bfloat16)
            y_base = cache[name]['output'].to(device).to(torch.float32) # float32 for stable MSE
            
            for mode in modes:
                if mode == "fp32" or mode == "bf16":
                    results_mse[mode].append(0.0)
                    continue
                    
                w_mode, a_mode = _resolve_modes(mode)
                
                # Quantize weight
                w_q, _ = _quantise_weight(orig_weight, w_mode, block_size)
                w_q = w_q.to(orig_weight.dtype)
                
                # Quantize activation
                x_q, _ = _quantise_activation(x_in.float(), a_mode, block_size)
                x_q = x_q.to(x_in.dtype)
                
                # Forward
                import torch.nn.functional as F
                if isinstance(layer, nn.Linear):
                    bias = layer.bias.to(x_q.dtype) if layer.bias is not None else None
                    y_q = F.linear(x_q, w_q, bias)
                else: # Conv1D
                    w_t = w_q.t().contiguous()
                    bias = layer.bias.to(x_q.dtype) if layer.bias is not None else None
                    size_out = x_q.shape[:-1] + (w_t.shape[1],)
                    if bias is not None:
                        y_q = torch.addmm(bias, x_q.view(-1, x_q.shape[-1]), w_t)
                    else:
                        y_q = x_q.view(-1, x_q.shape[-1]) @ w_t
                    y_q = y_q.view(size_out)
                    
                mse = F.mse_loss(y_q.float(), y_base).item()
                results_mse[mode].append(mse)
                
            # Free memory
            del x_in
            del y_base
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 3. Plotting
    print("Generating chart...")
    os.makedirs("results", exist_ok=True)
    
    x = np.arange(len(target_layer_keys))
    width = 0.8 / len(modes)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, mode in enumerate(modes):
        offset = (i - len(modes)/2 + 0.5) * width
        ax.bar(x + offset, results_mse[mode], width, label=mode, color=colors[i % len(colors)])
        
    ax.set_ylabel('Mean Squared Error (MSE)')
    ax.set_title(f'Layer-wise Output Reconstruction Error ({model_name})')
    ax.set_xticks(x)
    ax.set_xticklabels([k.split('.')[-1] + f"\n({k.split('.')[2]})" if len(k.split('.')) > 2 else k for k in target_layer_keys], rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    chart_path = "results/layer_diagnostics.png"
    plt.savefig(chart_path, dpi=300)
    print(f"Chart saved to {chart_path}")
    
    # Save JSON data
    data_path = "results/layer_diagnostics.json"
    with open(data_path, "w") as f:
        json.dump({"layers": target_layer_keys, "mse": results_mse}, f, indent=2)
    print(f"Data saved to {data_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    run_layer_diagnostics(cfg)
