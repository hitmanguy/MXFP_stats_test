"""
frameworks/speech.py
====================
LibriSpeech WER evaluation harness for Wav2Vec2.

Model:   facebook/wav2vec2-base-960h
Dataset: LibriSpeech test-clean  (HuggingFace: librispeech_asr, clean config, test split)
Metric:  Word Error Rate (WER %) — greedy CTC argmax decoding, no LM rescoring.

Documented reference (HF model card, greedy):
    test-clean WER = 3.4 %

Architecture notes for quantisation:
  - Feature extractor:  7 × nn.Conv1d layers  (operate on raw 16 kHz waveform)
  - Transformer:        12 × transformer blocks with nn.Linear projections
  - CTC head:           lm_head  (nn.Linear, vocab projection, equiv. of GPT-2 lm_head)

Layer interception:
  - nn.Conv1d  →  FakeQuantConv1d    (feature extractor convolutions)
  - nn.Linear  →  FakeQuantLinear    (transformer projections + lm_head)

  NOTE: HuggingFace's GPT-2 Conv1D (transformers.pytorch_utils.Conv1D) does NOT
  appear in Wav2Vec2 — this model uses plain nn.Conv1d and nn.Linear throughout.
  FakeQuantGPT2Conv1D is therefore NOT used here.
"""

from __future__ import annotations

# ── SSL cert fix for Windows environments ────────────────────────────────────
import os as _os
try:
    import certifi as _certifi
    _os.environ.setdefault("SSL_CERT_FILE", _certifi.where())
    _os.environ.setdefault("REQUESTS_CA_BUNDLE", _certifi.where())
except ImportError:
    pass
# ─────────────────────────────────────────────────────────────────────────────

import re
import os
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# WER helpers
# ─────────────────────────────────────────────────────────────────────────────

def _levenshtein(s1: list, s2: list) -> int:
    """Levenshtein edit distance between two sequences (words)."""
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            tmp = dp[j]
            dp[j] = prev if s1[i - 1] == s2[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = tmp
    return dp[n]


def wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate between a reference and hypothesis string."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    return _levenshtein(ref_words, hyp_words) / len(ref_words)


def corpus_wer(
    references: List[str],
    hypotheses: List[str],
) -> float:
    """
    Corpus-level WER: sum(edits) / sum(ref_words).
    This matches the standard corpus WER definition (not mean of per-utterance WERs).
    """
    total_edits = 0
    total_ref_words = 0
    for ref, hyp in zip(references, hypotheses):
        ref_words = ref.split()
        hyp_words = hyp.split()
        total_edits += _levenshtein(ref_words, hyp_words)
        total_ref_words += len(ref_words)
    if total_ref_words == 0:
        return 0.0
    return total_edits / total_ref_words * 100.0   # return as percentage


# ─────────────────────────────────────────────────────────────────────────────
# CTC greedy decoder
# ─────────────────────────────────────────────────────────────────────────────

def _ctc_greedy_decode(logits: torch.Tensor, processor) -> str:
    """
    Greedy CTC decode: argmax per frame → collapse repeated tokens → remove
    blank (pad_token_id) → decode token IDs to text.

    Args:
        logits: [batch=1, time, vocab] float tensor
        processor: Wav2Vec2 processor

    Returns:
        Decoded string (upper-case, no trailing whitespace).
    """
    # Argmax over vocab dimension: [batch, time]
    pred_ids = logits.argmax(dim=-1)
    
    # Let the processor handle CTC collapse and decoding
    text = processor.batch_decode(pred_ids)[0]
    return text.strip().upper()


# ─────────────────────────────────────────────────────────────────────────────
# Mode resolver (shared with language.py)
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_modes(quant_mode: str) -> Tuple[str, str]:
    from frameworks.language import _resolve_modes as _lang_resolve
    return _lang_resolve(quant_mode)


# ─────────────────────────────────────────────────────────────────────────────
# Layer substitution
# ─────────────────────────────────────────────────────────────────────────────

# Names to skip by default, analogous to GPT-2's lm_head skip policy.
# "lm_head" in Wav2Vec2 is the final CTC projection (nn.Linear, vocab size).
# "feature_extractor" covers the raw-waveform conv stack.
_DEFAULT_SKIP_NAMES = ["lm_head"]


def _replace_speech_layers(
    model: nn.Module,
    weight_mode: str,
    act_mode: str,
    block_size: int = 32,
    skip_names: Optional[List[str]] = None,
    skip_feature_extractor: bool = False,
) -> nn.Module:
    """
    Walk model and replace nn.Conv1d and nn.Linear with FakeQuant equivalents.

    skip_names:              list of dotted full-path names to leave unchanged.
    skip_feature_extractor:  if True, skip ALL submodules rooted at
                             'wav2vec2.feature_extractor' (the 7-layer conv stack).
    """
    from core.layers import FakeQuantLinear, FakeQuantConv1d

    if skip_names is None:
        skip_names = list(_DEFAULT_SKIP_NAMES)
    skip_set = set(skip_names)

    def _recurse(module: nn.Module, prefix: str = "") -> None:
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}".lstrip(".")

            # Skip entire subtree if it's the feature extractor and flag is set
            if skip_feature_extractor and full_name.startswith("wav2vec2.feature_extractor"):
                continue

            # Skip explicitly named layers
            if full_name in skip_set:
                continue

            if isinstance(child, nn.Conv1d):
                setattr(
                    module, name,
                    FakeQuantConv1d.from_conv1d(child, weight_mode, act_mode, block_size),
                )
            elif isinstance(child, nn.Linear):
                setattr(
                    module, name,
                    FakeQuantLinear.from_linear(child, weight_mode, act_mode, block_size),
                )
            else:
                _recurse(child, full_name)

    _recurse(model)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_librispeech_samples(
    n_samples: int = 200,
    seed: int = 42,
    split: str = "test",
) -> list:
    """
    Load n_samples utterances from LibriSpeech test-clean.
    Uses seeded random selection (same philosophy as language.py's chunk selection).
    """
    from datasets import load_dataset

    ds = load_dataset(
        "librispeech_asr",
        "clean",
        split=split,
        trust_remote_code=True,
        streaming=True,
    )

    # In streaming mode, we can't use len() or random indexing easily.
    # We will just take the first n_samples deterministically.
    selected = []
    for i, sample in enumerate(ds):
        if i >= n_samples:
            break
        selected.append(sample)
    
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Inference loop
# ─────────────────────────────────────────────────────────────────────────────

def _run_inference(
    model: nn.Module,
    processor,
    samples: list,
    device: torch.device,
) -> Tuple[List[str], List[str]]:
    """
    Run greedy CTC inference over a list of LibriSpeech samples.

    Returns (references, hypotheses) — both upper-cased.
    """
    model.eval()
    model.to(device)

    references: List[str] = []
    hypotheses: List[str] = []

    tokenizer = processor.tokenizer

    with torch.no_grad():
        for i, sample in enumerate(samples):
            audio_array = sample["audio"]["array"]
            sr = sample["audio"]["sampling_rate"]
            reference = sample["text"].strip().upper()

            # Safety check: model requires 16 kHz
            if sr != 16_000:
                # Resample if needed (should not happen for librispeech_asr)
                import warnings
                warnings.warn(
                    f"Sample {i} has sr={sr}, expected 16000. Skipping.",
                    RuntimeWarning,
                )
                continue

            # Feature extraction
            inputs = processor(
                audio_array,
                sampling_rate=16_000,
                return_tensors="pt",
                padding=False,
            )
            # Match model dtype for bf16 support
            model_dtype = next(model.parameters()).dtype
            input_values = inputs.input_values.to(device, dtype=model_dtype)

            # Forward pass
            with torch.autocast(device_type=device.type,
                                dtype=torch.float32,
                                enabled=False):
                outputs = model(input_values)

            logits = outputs.logits.float().cpu()

            # Greedy CTC decode
            hypothesis = _ctc_greedy_decode(logits, processor)

            references.append(reference)
            hypotheses.append(hypothesis)

            if (i + 1) % 50 == 0:
                running_wer = corpus_wer(references, hypotheses)
                print(f"    [{i+1}/{len(samples)}] running WER = {running_wer:.2f}%")

    return references, hypotheses


# ─────────────────────────────────────────────────────────────────────────────
# Main harness
# ─────────────────────────────────────────────────────────────────────────────

class SpeechEvalHarness:
    """
    LibriSpeech WER evaluation harness for Wav2Vec2.

    Args:
        model_name:              HF model ID (default: facebook/wav2vec2-base-960h)
        quant_mode:              Quantisation mode string
        n_samples:               Number of utterances to evaluate (default 200)
        seed:                    Random seed for sample selection (default 42)
        block_size:              MXFP4/MXFP8 block size (default 32)
        device:                  torch.device or None for auto
        skip_feature_extractor:  If True, leave conv feature extractor unquantised
        extra_skip_names:        Additional dotted-path names to skip
    """

    DOCUMENTED_REFERENCE_WER = 3.4   # % — HF model card, greedy decoding, test-clean

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base-960h",
        quant_mode: str = "fp32",
        n_samples: int = 200,
        seed: int = 42,
        block_size: int = 32,
        device: Optional[torch.device] = None,
        skip_feature_extractor: bool = False,
        extra_skip_names: Optional[List[str]] = None,
    ):
        self.model_name = model_name
        self.quant_mode = quant_mode
        self.n_samples = n_samples
        self.seed = seed
        self.block_size = block_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.skip_feature_extractor = skip_feature_extractor
        self.extra_skip_names = extra_skip_names or []

    def _load_model_and_processor(self):
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        model = Wav2Vec2ForCTC.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
        )
        return model, processor

    def run(self) -> Dict[str, Any]:
        """
        Run evaluation. Returns a dict with:
          wer_pct, references, hypotheses, quant_mode, n_samples,
          model_name, seed, skip_feature_extractor
        """
        weight_mode, act_mode = _resolve_modes(self.quant_mode)

        # Build skip list
        skip_names = list(_DEFAULT_SKIP_NAMES) + list(self.extra_skip_names)

        print(f"\n{'='*65}")
        print(f"  Model:      {self.model_name}")
        print(f"  Mode:       {self.quant_mode}  (w={weight_mode}, a={act_mode})")
        print(f"  Samples:    {self.n_samples}  |  Seed: {self.seed}")
        print(f"  Device:     {self.device}")
        print(f"  Skip FE:    {self.skip_feature_extractor}  |  Skip names: {skip_names}")
        print(f"{'='*65}")

        # Load model
        print("  Loading model and processor...")
        model, processor = self._load_model_and_processor()

        # Apply quantisation
        if self.quant_mode not in ("fp32", "bf16"):
            print("  Applying layer substitution...")
            model = _replace_speech_layers(
                model,
                weight_mode=weight_mode,
                act_mode=act_mode,
                block_size=self.block_size,
                skip_names=skip_names,
                skip_feature_extractor=self.skip_feature_extractor,
            )
            print("  Layer substitution complete.")
        elif self.quant_mode == "bf16":
            print("  Casting model to bfloat16...")
            model = model.to(torch.bfloat16)
            print("  Casting complete.")

        # Load data
        print(f"  Loading {self.n_samples} LibriSpeech test-clean samples...")
        samples = _load_librispeech_samples(
            n_samples=self.n_samples,
            seed=self.seed,
            split="test",
        )
        print(f"  Loaded {len(samples)} samples.")

        # Evaluate
        print("  Evaluating WER (greedy CTC)...")
        references, hypotheses = _run_inference(model, processor, samples, self.device)

        wer_pct = corpus_wer(references, hypotheses)

        from core.quantizer import bits_per_value
        eff_bits = bits_per_value(self.quant_mode)

        print(f"\n  > WER = {wer_pct:.2f}%  (eff_bits={eff_bits:.2f})")
        print(f"    Reference (documented greedy): {self.DOCUMENTED_REFERENCE_WER:.1f}%")
        delta = wer_pct - self.DOCUMENTED_REFERENCE_WER
        print(f"    Δ from reference: {delta:+.2f} pp")
        print()

        return {
            "wer_pct": wer_pct,
            "references": references,
            "hypotheses": hypotheses,
            "model_name": self.model_name,
            "quant_mode": self.quant_mode,
            "weight_mode": weight_mode,
            "act_mode": act_mode,
            "n_samples": len(references),
            "seed": self.seed,
            "skip_feature_extractor": self.skip_feature_extractor,
            "extra_skip_names": self.extra_skip_names,
            "eff_bits": eff_bits,
        }
