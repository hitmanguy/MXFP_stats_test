"""
analyze_results.py
==================
Pareto frontier analysis with:
  - Paired t-test
  - Wilcoxon signed-rank test
  - 95% bootstrap CI (10k resamples) on mean NLL difference
  - Cohen's d effect size
  - Pareto scatter plots (bits/value vs PPL)

Usage:
    python analyze_results.py                      # uses results/eval_ledger.db
    python analyze_results.py --db path/to/db      # custom DB path
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Statistical tests
# ─────────────────────────────────────────────────────────────────────────────

def paired_ttest(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """Returns (t_stat, p_value) for paired t-test of a vs b."""
    from scipy import stats
    result = stats.ttest_rel(a, b)
    return result.statistic, result.pvalue


def wilcoxon_test(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """Returns (stat, p_value) for Wilcoxon signed-rank test."""
    from scipy import stats
    diff = a - b
    if np.all(diff == 0):
        return 0.0, 1.0
    result = stats.wilcoxon(diff)
    return result.statistic, result.pvalue


def bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Bootstrap 95% CI on mean(a - b).
    Returns (mean_diff, ci_low, ci_high).
    """
    rng = np.random.default_rng(seed)
    n = len(a)
    diff = a - b
    mean_diff = diff.mean()

    boot_means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = diff[idx].mean()

    alpha = (1 - confidence) / 2
    ci_low = np.percentile(boot_means, 100 * alpha)
    ci_high = np.percentile(boot_means, 100 * (1 - alpha))
    return mean_diff, ci_low, ci_high


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for paired data: mean(diff) / std(diff)."""
    diff = a - b
    std = diff.std(ddof=1)
    if std == 0:
        return 0.0
    return diff.mean() / std


def full_significance_report(
    name_a: str,
    a: np.ndarray,
    name_b: str,
    b: np.ndarray,
) -> Dict:
    """Run all four tests and return a result dict."""
    t_stat, t_p = paired_ttest(a, b)
    w_stat, w_p = wilcoxon_test(a, b)
    mean_diff, ci_low, ci_high = bootstrap_ci(a, b)
    d = cohens_d(a, b)

    report = {
        "variant_a": name_a,
        "variant_b": name_b,
        "n": len(a),
        "mean_nll_a": float(np.mean(a)),
        "mean_nll_b": float(np.mean(b)),
        "mean_diff_a_minus_b": float(mean_diff),
        "paired_ttest": {"t_stat": t_stat, "p_value": t_p},
        "wilcoxon": {"stat": w_stat, "p_value": w_p},
        "bootstrap_95ci": {"ci_low": ci_low, "ci_high": ci_high},
        "cohens_d": d,
    }
    return report


def print_significance_report(report: Dict) -> None:
    a, b = report["variant_a"], report["variant_b"]
    print(f"\n{'─'*60}")
    print(f"Significance: [{a}] vs [{b}]  (n={report['n']})")
    print(f"  Mean NLL  {a}: {report['mean_nll_a']:.4f}")
    print(f"  Mean NLL  {b}: {report['mean_nll_b']:.4f}")
    print(f"  Mean diff (a−b):  {report['mean_diff_a_minus_b']:+.4f}")
    t = report["paired_ttest"]
    print(f"  Paired t-test:    t={t['t_stat']:.4f}  p={t['p_value']:.4e}")
    w = report["wilcoxon"]
    print(f"  Wilcoxon SR:      W={w['stat']:.1f}    p={w['p_value']:.4e}")
    ci = report["bootstrap_95ci"]
    print(f"  Bootstrap 95% CI: [{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}]")
    print(f"  Cohen's d:        {report['cohens_d']:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Pareto plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_pareto(
    results: List[Dict],
    output_path: str = "results/pareto_frontier.png",
) -> None:
    """Generate a Pareto scatter plot: bits/value vs PPL."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not available — skipping plot")
        return

    if not results:
        return

    bits = [r["eff_bits"] for r in results]
    ppls = [r["metric_value"] for r in results]
    labels = [r["quant_mode"] for r in results]

    plt.figure(figsize=(10, 6))
    sns.set_style("darkgrid")
    palette = sns.color_palette("husl", len(results))

    for i, (b, p, l) in enumerate(zip(bits, ppls, labels)):
        plt.scatter(b, p, color=palette[i], s=120, zorder=5, label=l)
        plt.annotate(l, (b, p), textcoords="offset points",
                     xytext=(6, 4), fontsize=8, color=palette[i])

    plt.xlabel("Effective Bits / Value", fontsize=12)
    plt.ylabel("Perplexity (↓ better)", fontsize=12)
    plt.title("MXFP4 Quantisation: Pareto Frontier — Bits/Value vs Perplexity", fontsize=13)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"  Pareto plot saved → {output_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# DB query helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_results_from_db(db_path: str) -> List[Dict]:
    import sqlite3
    conn = sqlite3.connect(db_path)
    cur = conn.execute(
        "SELECT model_family, quant_mode, seed, metric_name, metric_value, "
        "eff_bits, trigger_stats FROM evaluations ORDER BY eff_bits"
    )
    rows = []
    for r in cur.fetchall():
        rows.append({
            "model_family": r[0],
            "quant_mode": r[1],
            "seed": r[2],
            "metric_name": r[3],
            "metric_value": r[4],
            "eff_bits": r[5],
            "trigger_stats": json.loads(r[6]) if r[6] else {},
        })
    conn.close()
    return rows


def print_markdown_table(results: List[Dict]) -> None:
    ppl_results = [r for r in results if r["metric_name"] == "ppl"]
    if not ppl_results:
        print("No PPL results found.")
        return

    print("\n| Model | Mode | Bits/Val | PPL |")
    print("|-------|------|----------|-----|")
    for r in ppl_results:
        print(
            f"| {r['model_family']} | {r['quant_mode']} "
            f"| {r['eff_bits']:.2f} | {r['metric_value']:.2f} |"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze MXFP4 evaluation results")
    parser.add_argument("--db", default="results/eval_ledger.db", help="SQLite DB path")
    parser.add_argument("--plot", default="results/pareto_frontier.png", help="Pareto plot output")
    args = parser.parse_args()

    if not Path(args.db).exists():
        print(f"DB not found: {args.db}")
        return

    results = load_results_from_db(args.db)
    ppl_results = [r for r in results if r["metric_name"] == "ppl"]

    print_markdown_table(ppl_results)
    plot_pareto(ppl_results, args.plot)

    # Run significance tests on adjacent Pareto pairs (sorted by eff_bits)
    sorted_results = sorted(ppl_results, key=lambda r: r["eff_bits"])
    # For demo, we'd need per_chunk_nll stored in DB — this requires future schema extension
    print("\n[NOTE] Per-chunk NLL significance tests require per-chunk data stored in DB.")
    print("       Re-run with run_sweep.py --significance to compute paired tests in-memory.")


if __name__ == "__main__":
    main()
