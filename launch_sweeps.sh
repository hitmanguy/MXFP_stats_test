#!/bin/bash
# launch_sweeps.sh
# ─────────────────
# Automated execution launcher.
# IMPORTANT: Run inside the mxfp conda environment:
#   conda activate mxfp && bash launch_sweeps.sh

set -e

echo "======================================================"
echo "  MXFP4 Evaluation Framework — Full Launch"
echo "======================================================"

# Step 1: Unit tests
echo ""
echo "--- Step 1: Unit Tests ---"
python tests/test_quantizer.py

# Step 2: Acceptance gate (Part 3)
echo ""
echo "--- Step 2: Acceptance Gate (Part 3) ---"
python run_sweep.py --config configs/acceptance_gate.yaml

# Step 3: Full adaptive sweep (Part 3 + 4) with significance tests
echo ""
echo "--- Step 3: Full Adaptive Sweep + Significance Tests (Parts 3-4) ---"
python run_sweep.py --config configs/adaptive_residual_sweep.yaml --significance

# Step 4: Generate Pareto plots
echo ""
echo "--- Step 4: Pareto Analysis ---"
python analyze_results.py

echo ""
echo "======================================================"
echo "  All sweeps complete. Check results/ for outputs."
echo "======================================================"
