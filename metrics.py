"""
metrics.py — Evaluation metrics for SVD-Benchmark.

Implements Section 3.3 (Evaluation Metrics) of the paper:
  "A CWE-Aware Benchmark and Comparative Evaluation of LLMs for Java
   Software Vulnerability Detection"

Metrics:
  - Precision, Recall, F1-score, Accuracy  (standard definitions)
  - Binary Detection Rate    : recall on the vulnerable class;
                               any CWE prediction counts as a positive
  - CWE-Specific Detection Rate: recall requiring exact CWE-ID match

Two evaluation views (Section 3.3):
  Binary view      — measures whether a snippet was flagged as vulnerable
                     at all (ignores the specific CWE predicted)
  CWE-specific view— requires the predicted CWE-ID to exactly match the
                     ground-truth label for a TP

Output regex used for label-collapsing (Section 3.3):
  ^(BENIGN|CWE-\\d{2,4})$
  Non-matching responses are pre-collapsed to BENIGN before metrics are
  computed (done in evaluate_llm.py / evaluate_slm.py).
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helper: binary view label conversion
# ---------------------------------------------------------------------------
def to_binary(label: str) -> int:
    """Return 1 if label is any CWE (vulnerable), 0 if BENIGN."""
    return 0 if label.strip().upper() == "BENIGN" else 1


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------
def compute_confusion(
    ground_truths: list[str],
    predictions:   list[str],
    binary: bool = True,
) -> dict:
    """
    Compute TP, FP, TN, FN.

    If binary=True (Binary view): any CWE prediction vs BENIGN.
    If binary=False (CWE-specific view): exact label match required.
    """
    tp = fp = tn = fn = 0
    for gt, pred in zip(ground_truths, predictions):
        if binary:
            gt_b   = to_binary(gt)
            pred_b = to_binary(pred)
        else:
            gt_b   = 1 if gt   != "BENIGN" else 0
            pred_b = 1 if pred == gt else 0   # TP only on exact CWE match

        if binary:
            if   gt_b == 1 and pred_b == 1: tp += 1
            elif gt_b == 0 and pred_b == 1: fp += 1
            elif gt_b == 0 and pred_b == 0: tn += 1
            else:                           fn += 1
        else:
            # CWE-specific: TP = exact match; FN = vuln but wrong/missed
            if gt != "BENIGN":
                if pred == gt:  tp += 1
                else:           fn += 1
            else:
                if pred == "BENIGN": tn += 1
                else:                fp += 1

    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}


def precision(tp: int, fp: int) -> float:
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(tp: int, fn: int) -> float:
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1(prec: float, rec: float) -> float:
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def accuracy(tp: int, fp: int, tn: int, fn: int) -> float:
    total = tp + fp + tn + fn
    return (tp + tn) / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Full metric report
# ---------------------------------------------------------------------------
def compute_metrics(
    ground_truths: list[str],
    predictions:   list[str],
) -> dict:
    """
    Return a complete metrics dict for a set of predictions.

    Includes both Binary view and CWE-specific view results.
    """
    # --- Binary view ---
    conf_b = compute_confusion(ground_truths, predictions, binary=True)
    prec_b = precision(conf_b["TP"], conf_b["FP"])
    rec_b  = recall(conf_b["TP"],   conf_b["FN"])
    f1_b   = f1(prec_b, rec_b)
    acc_b  = accuracy(conf_b["TP"], conf_b["FP"], conf_b["TN"], conf_b["FN"])

    # --- CWE-specific view ---
    conf_c = compute_confusion(ground_truths, predictions, binary=False)
    prec_c = precision(conf_c["TP"], conf_c["FP"])
    rec_c  = recall(conf_c["TP"],   conf_c["FN"])
    f1_c   = f1(prec_c, rec_c)
    acc_c  = accuracy(conf_c["TP"], conf_c["FP"], conf_c["TN"], conf_c["FN"])

    return {
        "binary": {
            "precision":           round(prec_b * 100, 2),
            "recall":              round(rec_b  * 100, 2),
            "f1":                  round(f1_b   * 100, 2),
            "accuracy":            round(acc_b  * 100, 2),
            "detection_rate":      round(rec_b  * 100, 2),  # = recall on vuln class
            **conf_b,
        },
        "cwe_specific": {
            "precision":           round(prec_c * 100, 2),
            "recall":              round(rec_c  * 100, 2),
            "f1":                  round(f1_c   * 100, 2),
            "accuracy":            round(acc_c  * 100, 2),
            "detection_rate":      round(rec_c  * 100, 2),
            **conf_c,
        },
    }


# ---------------------------------------------------------------------------
# Per-CWE breakdown
# ---------------------------------------------------------------------------
def compute_per_cwe_metrics(
    ground_truths: list[str],
    predictions:   list[str],
) -> dict[str, dict]:
    """
    Return precision / recall / F1 for each CWE category.

    Used to identify CWE-level blind spots (Section 4.5 of the paper).
    """
    cwe_tp: dict[str, int] = defaultdict(int)
    cwe_fn: dict[str, int] = defaultdict(int)
    cwe_fp: dict[str, int] = defaultdict(int)

    for gt, pred in zip(ground_truths, predictions):
        if gt != "BENIGN":
            if pred == gt:
                cwe_tp[gt] += 1
            else:
                cwe_fn[gt] += 1
                if pred != "BENIGN":
                    cwe_fp[pred] += 1

    results = {}
    for cwe in sorted(set(list(cwe_tp) + list(cwe_fn))):
        tp = cwe_tp[cwe]
        fn = cwe_fn[cwe]
        fp = cwe_fp.get(cwe, 0)
        p  = precision(tp, fp)
        r  = recall(tp, fn)
        results[cwe] = {
            "precision": round(p * 100, 2),
            "recall":    round(r * 100, 2),
            "f1":        round(f1(p, r) * 100, 2),
            "support":   tp + fn,
        }
    return results


# ---------------------------------------------------------------------------
# Load predictions helper
# ---------------------------------------------------------------------------
def load_predictions(json_path: str) -> tuple[list[str], list[str]]:
    """Load a results JSON file and return (ground_truths, predictions)."""
    import json
    with open(json_path) as f:
        records = json.load(f)
    ground_truths = [r["ground_truth"] for r in records]
    predictions   = [r["prediction"]   for r in records]
    return ground_truths, predictions


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Compute metrics from a results JSON file.")
    parser.add_argument("results_file", help="Path to a model results JSON file")
    args = parser.parse_args()

    gts, preds = load_predictions(args.results_file)
    report     = compute_metrics(gts, preds)

    print("\n=== Binary View ===")
    for k, v in report["binary"].items():
        print(f"  {k:20s}: {v}")

    print("\n=== CWE-Specific View ===")
    for k, v in report["cwe_specific"].items():
        print(f"  {k:20s}: {v}")

    print("\n=== Per-CWE Breakdown ===")
    per_cwe = compute_per_cwe_metrics(gts, preds)
    for cwe, m in per_cwe.items():
        print(f"  {cwe}: F1={m['f1']}%  Precision={m['precision']}%  "
              f"Recall={m['recall']}%  Support={m['support']}")
