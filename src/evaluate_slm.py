"""
evaluate_slm.py — Open-source SLM inference via Ollama + RAG.

Models (Section 3.2, Table 2):
  - CodeGemma 2B     (Google DeepMind)  — ollama tag: codegemma:2b
  - Qwen2.5-Coder 1.5B (Alibaba)       — ollama tag: qwen2.5-coder:1.5b
  - Llama 3.2 1B     (Meta)            — ollama tag: llama3.2:1b

All models deployed locally using Ollama on Apple MacBook Pro (M3 Pro,
18 GB RAM) to verify consumer-grade hardware feasibility (Section 3.2).

Strategies applied to SLMs (Section 3.4):
  - One-Shot   : instruction + target code, no examples
  - Few-Shot ICL: instruction + k=3 examples (seed=42) + target code
  - RAG        : instruction + top-k SimCSE-retrieved exemplars + target code

Inference settings:
  - temperature=0  (deterministic — set via Ollama options)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import ollama
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATASET_PATH, RESULTS_DIR, FEW_SHOT_K, FEW_SHOT_SEED,
    TEMPERATURE, SLM_MODELS,
)
from data_utils import load_dataset, get_eval_set, get_rag_pool, sample_few_shot_examples
from prompts import build_one_shot_prompt, build_few_shot_prompt, build_rag_prompt
from evaluate_llm import parse_response   # reuse the same output-parsing logic
from rag import SimCSERetriever


# ---------------------------------------------------------------------------
# Ollama inference
# ---------------------------------------------------------------------------
def call_ollama(prompt: str, model_tag: str) -> str:
    """Call a locally running Ollama model with temperature=0."""
    response = ollama.generate(
        model=model_tag,
        prompt=prompt,
        options={"temperature": TEMPERATURE, "num_predict": 20},
    )
    return response.get("response", "")


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
def evaluate(
    model_key: str,
    strategy: str,
    df: pd.DataFrame,
    retriever: SimCSERetriever | None = None,
) -> list[dict]:
    """
    Run inference for a single (SLM, strategy) combination.

    For RAG strategy, a pre-built SimCSERetriever must be supplied.
    """
    cfg       = SLM_MODELS[model_key]
    model_tag = cfg["ollama_tag"]

    # Sample few-shot examples once — fixed seed, same set for all instances
    few_shot_examples = (
        sample_few_shot_examples(df, k=FEW_SHOT_K, seed=FEW_SHOT_SEED)
        if strategy == "few_shot"
        else []
    )

    if strategy == "rag" and retriever is None:
        raise ValueError("RAG strategy requires a pre-built SimCSERetriever.")

    eval_set = get_eval_set(df)
    results  = []

    for idx, row in tqdm(
        eval_set.iterrows(),
        total=len(eval_set),
        desc=f"{cfg['paper_name']} / {strategy}",
    ):
        code  = row["Code Snippet"]
        label = row["label"]

        if strategy == "one_shot":
            prompt = build_one_shot_prompt(code)
        elif strategy == "few_shot":
            prompt = build_few_shot_prompt(code, few_shot_examples)
        elif strategy == "rag":
            retrieved = retriever.retrieve(code)
            prompt    = build_rag_prompt(code, retrieved)
        else:
            raise ValueError(f"Unknown strategy: {strategy!r}")

        raw        = call_ollama(prompt, model_tag)
        prediction = parse_response(raw)

        results.append({
            "index":        idx,
            "ground_truth": label,
            "prediction":   prediction,
            "raw_response": raw,
        })

    return results


def save_results(results: list[dict], model_key: str, strategy: str) -> Path:
    out_dir = Path(RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_key}_{strategy}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate SLMs on SVD-Benchmark (via Ollama).")
    parser.add_argument("--model",    choices=list(SLM_MODELS.keys()) + ["all"],
                        default="all")
    parser.add_argument("--strategy", choices=["one_shot", "few_shot", "rag", "all"],
                        default="all")
    args = parser.parse_args()

    df = load_dataset(DATASET_PATH)

    models     = list(SLM_MODELS.keys())            if args.model    == "all" else [args.model]
    strategies = ["one_shot", "few_shot", "rag"]    if args.strategy == "all" else [args.strategy]

    # Build the RAG index once and reuse across all models
    retriever = None
    if "rag" in strategies:
        pool      = get_rag_pool(df)
        retriever = SimCSERetriever()
        retriever.build_index(pool)

    for model_key in models:
        for strategy in strategies:
            print(f"\n=== {SLM_MODELS[model_key]['paper_name']} | {strategy} ===")
            results = evaluate(model_key, strategy, df, retriever=retriever)
            save_results(results, model_key, strategy)


if __name__ == "__main__":
    main()
