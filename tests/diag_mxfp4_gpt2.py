"""
tests/diag_mxfp4_gpt2.py
=========================
Diagnostic: isolate which component (weight quant, activation quant, lm_head)
is responsible for the PPL gap.

Tests (5 chunks, seed=42):
  fp32           → baseline
  mxfp4_w_only   → weight-only, no activation quant, no lm_head quant
  mxfp4_w_nolmh  → weight+act, skip lm_head
  mxfp4_both     → weight+act, include lm_head (current default)
"""

from __future__ import annotations
import os
try:
    import certifi as _c
    os.environ.setdefault("SSL_CERT_FILE", _c.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", _c.where())
except ImportError:
    pass

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch
import torch.nn as nn

# ── helpers ──────────────────────────────────────────────────────────────────

def load_gpt2_and_chunks(n_chunks=5, seed=42, seq_len=1024):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = 1_000_000_000
    model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32)

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    ids = tokenizer.encode(text)
    tokens = torch.tensor(ids, dtype=torch.long)
    n_avail = len(tokens) // seq_len
    chunks_all = [tokens[i*seq_len:(i+1)*seq_len] for i in range(n_avail)]
    torch.manual_seed(seed)
    perm = torch.randperm(len(chunks_all))
    chunks = [chunks_all[i] for i in perm[:n_chunks].tolist()]
    return model, chunks


def eval_ppl(model, chunks, device):
    model.eval().to(device)
    total_nll, total_tok = 0.0, 0
    with torch.no_grad():
        for ch in chunks:
            ids = ch.unsqueeze(0).to(device)
            loss = model(ids, labels=ids).loss.item()
            total_nll += loss * (ch.numel() - 1)
            total_tok += ch.numel() - 1
    return math.exp(total_nll / total_tok)


def replace_layers(model, weight_mode, act_mode, skip_names=None):
    """Replace Conv1D/Linear with FakeQuant versions, optionally skipping names."""
    from core.layers import FakeQuantLinear, FakeQuantGPT2Conv1D
    try:
        from transformers.pytorch_utils import Conv1D as HF_Conv1D
        has_c1d = True
    except ImportError:
        has_c1d = False

    skip_names = set(skip_names or [])

    def _recurse(module, prefix=""):
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}".lstrip(".")
            if full_name in skip_names:
                print(f"    [SKIP] {full_name}")
                continue
            if has_c1d and isinstance(child, HF_Conv1D):
                setattr(module, name,
                        FakeQuantGPT2Conv1D.from_conv1d(child, weight_mode, act_mode))
            elif isinstance(child, nn.Linear):
                setattr(module, name,
                        FakeQuantLinear.from_linear(child, weight_mode, act_mode))
            else:
                _recurse(child, full_name)

    _recurse(model)
    return model


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print("Loading model and 5 chunks (seed=42)...")
    base_model, chunks = load_gpt2_and_chunks(n_chunks=5, seed=42)
    print("Done.\n")

    scenarios = [
        ("fp32 baseline",            None,     None,       []),
        ("w=mxfp4, a=fp32 (weight-only, include lm_head)",
                                     "mxfp4",  "fp32",     []),
        ("w=mxfp4, a=fp32 (weight-only, SKIP lm_head)",
                                     "mxfp4",  "fp32",     ["lm_head"]),
        ("w=mxfp4, a=mxfp4 (SKIP lm_head)",
                                     "mxfp4",  "mxfp4",    ["lm_head"]),
        ("w=mxfp4, a=mxfp4 (include lm_head) [current default]",
                                     "mxfp4",  "mxfp4",    []),
        ("w=mxfp4_residual, a=mxfp4_residual (SKIP lm_head)",
                                     "mxfp4_residual", "mxfp4_residual", ["lm_head"]),
        ("w=mxfp4_residual, a=mxfp4_residual (include lm_head)",
                                     "mxfp4_residual", "mxfp4_residual", []),
    ]

    results = []
    for desc, wm, am, skip in scenarios:
        import copy
        m = copy.deepcopy(base_model)
        if wm is not None:
            m = replace_layers(m, wm, am, skip_names=skip)
        ppl = eval_ppl(m, chunks, device)
        print(f"  [{ppl:8.2f}]  {desc}")
        results.append((desc, ppl))
        del m

    print("\n\nSUMMARY:")
    print(f"  {'PPL':>8}  Description")
    print("  " + "─"*70)
    for desc, ppl in results:
        print(f"  {ppl:8.2f}  {desc}")
    print()
    print("Reference (50 chunks seed=42):")
    print("  fp32 baseline        → 29.98")
    print("  mxfp4 (both)         → 109.72")
    print("  mxfp4_residual both  → 30.13")


if __name__ == "__main__":
    main()
