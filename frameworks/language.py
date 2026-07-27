"""
frameworks/language.py
======================
GPT-2 / LLaMA / Mistral WikiText-2 perplexity evaluation harness.

Evaluation methodology (must match exactly for reproducible gate numbers):
  - Dataset: Salesforce/wikitext, wikitext-2-raw-v1, test split
  - Chunking: 1024-token chunks, 50 chunks selected via seeded random permutation
    (torch.randperm with manual_seed(seed)) — NOT first-50-sequential
  - Primary seed: 42; secondary: 99

Layer interception:
  - transformers.pytorch_utils.Conv1D  →  FakeQuantGPT2Conv1D
  - torch.nn.Linear                   →  FakeQuantLinear

Usage:
    from frameworks.language import LanguageEvalHarness
    h = LanguageEvalHarness("gpt2", quant_mode="mxfp4", seed=42)
    result = h.run()
    print(result["ppl"])
"""
from __future__ import annotations

# ── SSL cert fix for Windows environments ────────────────────────────────────
# Some Windows machines have a corrupted or incomplete cert store, causing
# aiohttp (a transitive dependency of datasets) to crash at import time with:
#   ssl.SSLError: [ASN1: NOT_ENOUGH_DATA]
# Fix: point Python's SSL to the certifi CA bundle instead of the Windows store.
import os as _os
try:
    import certifi as _certifi
    _os.environ.setdefault("SSL_CERT_FILE", _certifi.where())
    _os.environ.setdefault("REQUESTS_CA_BUNDLE", _certifi.where())
except ImportError:
    pass  # certifi not installed; hope the system certs work
# ─────────────────────────────────────────────────────────────────────────────

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Mode → (weight_mode, act_mode) decomposition
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_modes(quant_mode: str) -> Tuple[str, str]:
    """
    Map a high-level quant_mode string to (weight_mode, act_mode).

    Conventions:
      fp32                        → w=fp32,  a=fp32
      bf16                        → w=bf16,  a=bf16
      mxfp4                       → w=mxfp4, a=mxfp4
      mxfp4_residual / mxfp4_residual_full → w=mxfp4_residual, a=mxfp4_residual
      mxfp4_residual_act_only     → w=mxfp4, a=mxfp4_residual
      mxfp4_residual_weight_only  → w=mxfp4_residual, a=mxfp4
      mxfp4_adaptive_<X>          → w=mxfp4, a=mxfp4_adaptive_<X>
    """
    if quant_mode in ("fp32", "bf16"):
        return quant_mode, quant_mode
    if quant_mode in ("mxfp4", "nvfp4", "mxfp8_e4m3", "mxfp8_e5m2"):
        return quant_mode, quant_mode
    if quant_mode in ("mxfp4_residual", "mxfp4_residual_full",
                      "mxfp8_e4m3_residual", "mxfp8_e5m2_residual"):
        return quant_mode, quant_mode
    if quant_mode == "mxfp4_residual_act_only":
        return "mxfp4", "mxfp4_residual"
    if quant_mode == "mxfp4_residual_weight_only":
        return "mxfp4_residual", "mxfp4"
    if quant_mode.startswith("mxfp4_adaptive"):
        # adaptive applies to activations only; weights stay at naive mxfp4
        return "mxfp4", quant_mode
    if quant_mode.startswith("mxfp8_e4m3_adaptive"):
        return "mxfp8_e4m3", quant_mode
    if quant_mode.startswith("mxfp8_e5m2_adaptive"):
        return "mxfp8_e5m2", quant_mode
    return "mxfp4", "mxfp4"


# ─────────────────────────────────────────────────────────────────────────────
# Layer substitution
# ─────────────────────────────────────────────────────────────────────────────

def _replace_layers(
    model: nn.Module,
    weight_mode: str,
    act_mode: str,
    block_size: int = 32,
    skip_names: Optional[List[str]] = None,
) -> nn.Module:
    """
    Walk the model and replace Conv1D / Linear with their FakeQuant equivalents.
    Modifies model in-place, returns it.
    """
    from core.layers import FakeQuantLinear, FakeQuantGPT2Conv1D

    if skip_names is None:
        skip_names = ["lm_head"]
    skip_set = set(skip_names)

    try:
        from transformers.pytorch_utils import Conv1D as HF_Conv1D
        _HAS_CONV1D = True
    except ImportError:
        _HAS_CONV1D = False

    def _recurse(module: nn.Module, prefix: str = ""):
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}".lstrip(".")
            if full_name in skip_set:
                # Skip replacing this layer entirely
                continue
                
            if _HAS_CONV1D and isinstance(child, HF_Conv1D):
                setattr(
                    module, name,
                    FakeQuantGPT2Conv1D.from_conv1d(child, weight_mode, act_mode, block_size),
                )
            elif isinstance(child, nn.Linear):
                setattr(
                    module, name,
                    FakeQuantLinear.from_linear(child, weight_mode, act_mode, block_size),
                )
            else:
                # Recurse
                _recurse(child, full_name)

    _recurse(model)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Dataset + chunking
# ─────────────────────────────────────────────────────────────────────────────

def _load_wikitext_chunks(
    tokenizer,
    seq_len: int = 1024,
    n_chunks: int = 50,
    seed: int = 42,
) -> List[torch.Tensor]:
    """
    Load WikiText-2 test split and return n_chunks of seq_len tokens each,
    selected via seeded random permutation over ALL available chunks.

    The permutation approach (not sequential slicing) is the spec-correct
    method that produces the reference numbers.
    """
    from datasets import load_dataset

    # Suppress false tokenizer length warnings
    tokenizer.model_max_length = 1_000_000_000

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])

    # Tokenise full text at once
    token_ids = tokenizer.encode(text)
    token_tensor = torch.tensor(token_ids, dtype=torch.long)

    # Chunk into seq_len blocks
    total_tokens = token_tensor.numel()
    n_available = total_tokens // seq_len
    chunks = [token_tensor[i * seq_len: (i + 1) * seq_len] for i in range(n_available)]

    # Select n_chunks via seeded randperm (spec-correct)
    torch.manual_seed(seed)
    perm = torch.randperm(len(chunks))
    selected_indices = perm[:n_chunks].tolist()
    return [chunks[i] for i in selected_indices]


# ─────────────────────────────────────────────────────────────────────────────
# PPL computation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_ppl_on_chunks(
    model: nn.Module,
    chunks: List[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[float, List[float]]:
    """
    Compute perplexity on a list of fixed-length token chunks.
    Returns (ppl, per_chunk_nll_list).
    """
    model.eval()
    model.to(device)

    total_nll = 0.0
    total_tokens = 0
    per_chunk_nll: List[float] = []

    with torch.no_grad():
        for chunk in chunks:
            input_ids = chunk.unsqueeze(0).to(device)  # [1, seq_len]
            with torch.autocast(device_type=device.type,
                                dtype=dtype,
                                enabled=(dtype != torch.float32)):
                outputs = model(input_ids, labels=input_ids)
            # outputs.loss is mean NLL over the chunk (shifted by 1)
            nll = outputs.loss.item()
            per_chunk_nll.append(nll)
            total_nll += nll * (chunk.numel() - 1)   # weight by token count
            total_tokens += chunk.numel() - 1

    ppl = math.exp(total_nll / total_tokens) if total_tokens > 0 else float("inf")
    return ppl, per_chunk_nll


# ─────────────────────────────────────────────────────────────────────────────
# Main harness
# ─────────────────────────────────────────────────────────────────────────────

class LanguageEvalHarness:
    """
    Full language-model PPL evaluation harness.

    Args:
        model_name:  HF model ID (e.g. "gpt2")
        quant_mode:  Quantisation variant string
        seed:        Random seed for chunk selection (default 42)
        n_chunks:    Number of 1024-token chunks to evaluate (default 50)
        seq_len:     Chunk length in tokens (default 1024)
        block_size:  MXFP4 block size (default 32)
        device:      torch.device or None for auto
        hf_token:    HuggingFace token for gated models
    """

    GATED_MODELS = {
        "meta-llama/Meta-Llama-3-8B",
        "meta-llama/Llama-2-7b-hf",
        "mistralai/Mistral-7B-v0.3",
    }

    def __init__(
        self,
        model_name: str = "gpt2",
        quant_mode: str = "fp32",
        seed: int = 42,
        n_chunks: int = 50,
        seq_len: int = 1024,
        block_size: int = 32,
        device: Optional[torch.device] = None,
        hf_token: Optional[str] = None,
    ):
        self.model_name = model_name
        self.quant_mode = quant_mode
        self.seed = seed
        self.n_chunks = n_chunks
        self.seq_len = seq_len
        self.block_size = block_size
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Validate gated model access
        if model_name in self.GATED_MODELS:
            if not self.hf_token:
                raise EnvironmentError(
                    f"Model '{model_name}' is gated. "
                    "Set HF_TOKEN environment variable and pass it explicitly. "
                    "Do NOT fall back to a different model."
                )

    def _load_model_and_tokenizer(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        token_kwargs = {"token": self.hf_token} if self.hf_token else {}

        tokenizer = AutoTokenizer.from_pretrained(self.model_name, **token_kwargs)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,   # always load FP32; cast later if needed
            **token_kwargs,
        )
        return model, tokenizer

    def _get_dtype(self) -> torch.dtype:
        if self.quant_mode == "bf16":
            return torch.bfloat16
        return torch.float32

    def run(self) -> Dict[str, Any]:
        """
        Run evaluation. Returns a dict with:
          ppl, per_chunk_nll, quant_mode, seed, model_name,
          eff_bits, weight_mode, act_mode
        """
        weight_mode, act_mode = _resolve_modes(self.quant_mode)
        dtype = self._get_dtype()

        print(f"\n{'='*60}")
        print(f"  Model:      {self.model_name}")
        print(f"  Mode:       {self.quant_mode}  (w={weight_mode}, a={act_mode})")
        print(f"  Seed:       {self.seed}  |  Chunks: {self.n_chunks}  |  SeqLen: {self.seq_len}")
        print(f"  Device:     {self.device}  |  dtype: {dtype}")
        print(f"{'='*60}")

        # Load
        model, tokenizer = self._load_model_and_tokenizer()

        # Apply quant (skip for fp32/bf16 baselines)
        if self.quant_mode not in ("fp32", "bf16"):
            print("  Applying layer substitution...")
            model = _replace_layers(model, weight_mode, act_mode, self.block_size)
            print("  Layer substitution complete.")

        # bf16 cast
        if dtype == torch.bfloat16:
            model = model.to(torch.bfloat16)

        # Chunks
        print("  Loading WikiText-2 chunks...")
        chunks = _load_wikitext_chunks(tokenizer, self.seq_len, self.n_chunks, self.seed)
        print(f"  Loaded {len(chunks)} chunks.")

        # Evaluate
        print("  Evaluating PPL...")
        ppl, per_chunk_nll = _compute_ppl_on_chunks(model, chunks, self.device, dtype)

        from core.quantizer import bits_per_value
        eff_bits = bits_per_value(self.quant_mode)

        print(f"\n  ✓ PPL = {ppl:.4f}  (eff_bits={eff_bits:.2f})")
        print()

        return {
            "ppl": ppl,
            "per_chunk_nll": per_chunk_nll,
            "model_name": self.model_name,
            "quant_mode": self.quant_mode,
            "weight_mode": weight_mode,
            "act_mode": act_mode,
            "seed": self.seed,
            "n_chunks": self.n_chunks,
            "eff_bits": eff_bits,
        }
