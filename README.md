# MXFP4 Residual Quantisation Evaluation Framework

Pure-PyTorch multi-modal low-bit (MXFP4/NVFP4) quantisation framework with residual refinement, adaptive SQNR-triggered passes, and full statistical analysis.

## Environment Setup

**Every command must run inside the `mxfp` conda environment.**

```bash
# Activate — do this every time before running anything
conda activate mxfp

# First-time setup from scratch:
conda env create -f environment.yml
conda activate mxfp
```

## Running the Acceptance Gate (Part 3)

```bash
conda activate mxfp
python run_sweep.py --config configs/acceptance_gate.yaml
```

This runs GPT-2 on WikiText-2 (50 chunks, seed=42 & 99) across all 7 variants and prints a PASS/FAIL table against the reference numbers.

## Running the Full Adaptive Sweep

```bash
conda activate mxfp
python run_sweep.py --config configs/adaptive_residual_sweep.yaml --significance
```

## Quick CPU Smoke Test (fast, no GPU required)

```bash
conda activate mxfp
python run_sweep.py --config configs/debug_gpt2.yaml
```

## Analyzing Results

```bash
conda activate mxfp
python analyze_results.py
```

Generates:
- Markdown table from `results/eval_ledger.db`
- `results/pareto_frontier.png` Pareto scatter plot

## Reference Numbers (Part 3 Gate, GPT-2, seed=42, 50 chunks)

| Variant | Expected PPL | Tolerance |
|---|---|---|
| FP32 baseline | 29.98 | ±0.5 |
| BF16 baseline | 30.24 | ±0.5 |
| Naive MXFP4 | 109.72 | ±5 |
| Static full residual | 30.13 | ±1 |
| Static act-only residual | 35.33 | ±1.5 |
| Adaptive SQNR<15dB | 61.49 | ±3 |
| Adaptive SQNR<18dB | 34.87 | ±1.5 |

## Directory Structure

```
software_side_tests/
├── core/
│   ├── quantizer.py      # MXFP4/NVFP4 quantization math (OCP-spec-correct)
│   ├── layers.py         # FakeQuantLinear, FakeQuantGPT2Conv1D
│   └── metrics.py        # EvalLogger (SQLite + JSONL)
├── frameworks/
│   ├── language.py       # GPT-2/LLM WikiText-2 harness
│   ├── vision.py         # (Part 5) ImageNet harness
│   ├── speech.py         # (Part 5) LibriSpeech harness
│   └── recsys.py         # (Part 5) Criteo AUC harness
├── configs/
│   ├── acceptance_gate.yaml
│   ├── adaptive_residual_sweep.yaml
│   ├── debug_gpt2.yaml
│   └── mxfp4_vs_nvfp4_baselines.yaml
├── results/
│   ├── eval_ledger.db    # SQLite telemetry
│   └── eval_ledger.jsonl # Streaming JSON log
├── run_sweep.py          # Master CLI runner
├── analyze_results.py    # Pareto analysis + significance tests
├── environment.yml       # Reproducible conda environment
└── README.md
```

## Key Design Decisions

- **No custom CUDA kernels**: All quantization uses plain PyTorch tensor ops, runs identically on CPU/GPU.
- **GPT-2 Conv1D layout**: HuggingFace stores Conv1D weights as `[in, out]` (transposed vs `nn.Linear`). `FakeQuantGPT2Conv1D` transposes before blocking, quantizes, then transposes back.
- **OCP-spec E8M0 scale**: Uses `floor(log2(amax)) - floor(log2(FORMAT_MAX))` (not ceil) — the spec-correct rounding direction.
- **Round-to-nearest-even**: Tie midpoints (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0) resolve to the even-indexed neighbor, not always up or always down.
- **Seeded chunk selection**: Uses `torch.randperm(seed=42)` over all available chunks, NOT first-50-sequential.

## Gated Models (LLaMA/Mistral)

```bash
export HF_TOKEN="your_huggingface_token"
python run_sweep.py --config configs/llama_sweep.yaml
```

If `HF_TOKEN` is missing, the code errors loudly and does NOT fall back to a different model.
