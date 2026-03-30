"""
statistical_validation.py — Bootstrap CIs, McNemar's test, and Odds Ratio.

Implements Section 3.5 (Statistical Validation) of the paper:
  "A CWE-Aware Benchmark and Comparative Evaluation of LLMs for Java
   Software Vulnerability Detection"

Three complementary validation procedures:

1. Bootstrap Confidence Intervals (Section 3.5)
   - B = 5,000 resamples with replacement
   - 2.5th and 97.5th percentiles → 95% CI
   - Applied per model to bound metric uncertainty

2. McNemar's Test with Continuity Correction (Section 3.5)
   - For paired classifier comparisons on the same evaluation set
   - χ² = (|n₁₀ − n₀₁| − 1)² / (n₁₀ + n₀₁)
   - p < 0.05 → statistically significant difference
   - Applied to: ChatGPT-5.2 vs. CodeGemma,
                 ChatGPT-5.2 vs. Semgrep,
                 CodeGemma   vs. Semgrep

3. Odds Ratio (Section 3.5)
   - OR = n₁₀ / n₀₁  (McNemar discordant counts)
   - OR > 1 → model A correctly classifies more instances
              that model B misses than the reverse
   - Quantifies practical magnitude independent of sample size
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from metrics import compute_metrics, load_predictions, to_binary
from config import N_BOOTSTRAP


# ---------------------------------------------------------------------------
# 1. Bootstrap Confidence Intervals
# ---------------------------------------------------------------------------
def bootstrap_f1_ci(
    ground_truths: list[str],
    predictions:   list[str],
    n_bootstrap:   int = N_BOOTSTRAP,
    ci:            float = 0.95,
    seed:          int = 0,
) -> tuple[float, float, float]:
    """
    Non-parametric bootstrap CI for F1-score (binary view).

    Returns (f1_point_estimate, ci_lower, ci_upper) — all as percentages.

    The evaluation set (n = 5,054) is resampled with replacement B=5,000
    times; F1 is recomputed on every resample.  The 2.5th and 97.5th
    percentiles yield the 95% CI (Section 3.5).
    """
    rng   = np.random.default_rng(seed)
    n     = len(ground_truths)
    gts   = np.array(ground_truths)
    preds = np.array(predictions)

    f1_scores = []
    for _ in range(n_bootstrap):
        idx    = rng.integers(0, n, size=n)
        result = compute_metrics(gts[idx].tolist(), preds[idx].tolist())
        f1_scores.append(result["binary"]["f1"])

    f1_scores = np.array(f1_scores)
    alpha     = 1 - ci
    lower     = float(np.percentile(f1_scores, 100 * alpha / 2))
    upper     = float(np.percentile(f1_scores, 100 * (1 - alpha / 2)))

    # Point estimate from full data
    point = compute_metrics(ground_truths, predictions)["binary"]["f1"]
    return point, lower, upper


# ---------------------------------------------------------------------------
# 2. McNemar's Test with Continuity Correction
# ---------------------------------------------------------------------------
def mcnemar_test(
    ground_truths: list[str],
    predictions_a: list[str],
    predictions_b: list[str],
) -> dict:
    """
    McNemar's test (with continuity correction) for two classifiers on
    the same evaluation set (Section 3.5).

    Contingency table (binary view — correct = predicted label matches GT):
                     B correct   B wrong
      A correct   |    n₀₀    |   n₁₀  |
      A wrong     |    n₀₁    |   n₁₁  |

    Test statistic:
      χ² = (|n₁₀ − n₀₁| − 1)² / (n₁₀ + n₀₁)

    The −1 continuity correction is applied as specified in Section 3.5.

    Returns a dict with keys: n10, n01, chi2, p_value, odds_ratio.
    """
    n10 = n01 = 0   # discordant counts
    for gt, pa, pb in zip(ground_truths, predictions_a, predictions_b):
        gt_b = to_binary(gt)
        a_correct = (to_binary(pa) == gt_b)
        b_correct = (to_binary(pb) == gt_b)

        if a_correct and not b_correct:
            n10 += 1   # A correct, B wrong
        elif not a_correct and b_correct:
            n01 += 1   # A wrong, B correct

    discordant = n10 + n01
    if discordant == 0:
        return {"n10": 0, "n01": 0, "chi2": 0.0, "p_value": 1.0, "odds_ratio": float("nan")}

    # χ² with continuity correction (Equation in Section 3.5)
    chi2 = (abs(n10 - n01) - 1) ** 2 / discordant

    # p-value from chi-squared distribution with df=1
    from scipy.stats import chi2 as chi2_dist
    p_value = float(1 - chi2_dist.cdf(chi2, df=1))

    return {
        "n10":        n10,
        "n01":        n01,
        "chi2":       round(chi2,    3),
        "p_value":    p_value,
        "p_display":  "< 0.001" if p_value < 0.001 else f"{p_value:.4f}",
        "odds_ratio": odds_ratio(n10, n01),
    }


# ---------------------------------------------------------------------------
# 3. Odds Ratio
# ---------------------------------------------------------------------------
def odds_ratio(n10: int, n01: int) -> float:
    """
    Odds ratio = n₁₀ / n₀₁  (McNemar discordant counts, Section 3.5).

    OR > 1 means model A correctly classifies more instances that B
    misses than the reverse — A is practically superior.
    """
    if n01 == 0:
        return float("inf")
    return round(n10 / n01, 2)


# ---------------------------------------------------------------------------
# Full validation report
# ---------------------------------------------------------------------------
def full_report(
    ground_truths:  list[str],
    predictions_a:  list[str],
    predictions_b:  list[str],
    label_a:        str = "Model A",
    label_b:        str = "Model B",
) -> None:
    """Print bootstrap CIs and McNemar results for both models."""
    print(f"\n{'='*60}")
    print(f"Statistical Validation: {label_a}  vs  {label_b}")
    print(f"{'='*60}")

    for label, preds in [(label_a, predictions_a), (label_b, predictions_b)]:
        point, lo, hi = bootstrap_f1_ci(ground_truths, preds)
        print(f"\n{label}")
        print(f"  F1 = {point:.2f}%  95% CI [{lo:.1f}, {hi:.1f}]")

    mc = mcnemar_test(ground_truths, predictions_a, predictions_b)
    print(f"\nMcNemar's Test  ({label_a} vs {label_b})")
    print(f"  n₁₀ = {mc['n10']}   n₀₁ = {mc['n01']}")
    print(f"  χ²  = {mc['chi2']}")
    print(f"  p   = {mc['p_display']}")
    print(f"  OR  = {mc['odds_ratio']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap CI + McNemar's test for two prediction files."
    )
    parser.add_argument("--predictions_a", required=True, help="JSON results file for model A")
    parser.add_argument("--predictions_b", required=True, help="JSON results file for model B")
    parser.add_argument("--label_a", default="Model A")
    parser.add_argument("--label_b", default="Model B")
    parser.add_argument("--bootstrap_only", action="store_true",
                        help="Only compute bootstrap CI for model A (no pairwise test)")
    args = parser.parse_args()

    gts_a, preds_a = load_predictions(args.predictions_a)
    gts_b, preds_b = load_predictions(args.predictions_b)

    # Ground truths must match
    assert gts_a == gts_b, "Ground truth labels differ between the two files."

    if args.bootstrap_only:
        point, lo, hi = bootstrap_f1_ci(gts_a, preds_a)
        print(f"{args.label_a}: F1 = {point:.2f}%  95% CI [{lo:.1f}, {hi:.1f}]")
    else:
        full_report(gts_a, preds_a, preds_b, args.label_a, args.label_b)


if __name__ == "__main__":
    main()
