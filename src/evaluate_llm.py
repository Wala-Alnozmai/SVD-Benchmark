"""
evaluate_llm.py — Proprietary LLM inference for SVD-Benchmark.

Models (Section 3.2):
  - ChatGPT-5.2   (OpenAI API)
  - Claude 4.5 Sonnet  (Anthropic API)
  - Gemini 3.0 Flash   (Google AI API)

Strategies applied to proprietary LLMs (Section 3.4):
  - One-Shot    : instruction + target code, no examples
  - Few-Shot ICL: instruction + k=3 examples (seed=42) + target code

RAG is deliberately NOT applied to proprietary LLMs to isolate their
intrinsic semantic reasoning and mitigate data-contamination risks from
closed-source training corpora (Section 3.4).

Inference settings (Section 3.2):
  - temperature=0  (deterministic, reproducible)

Output parsing (Section 3.3):
  - Valid outputs match ^(BENIGN|CWE-\\d{2,4})$
  - Non-conforming responses are collapsed to BENIGN
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATASET_PATH, RESULTS_DIR, FEW_SHOT_K, FEW_SHOT_SEED,
    TEMPERATURE, LLM_MODELS, OUTPUT_REGEX,
)
from data_utils import load_dataset, get_eval_set, sample_few_shot_examples
from prompts import build_one_shot_prompt, build_few_shot_prompt


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------
_OUTPUT_RE = re.compile(OUTPUT_REGEX)


def parse_response(raw: str) -> str:
    """
    Extract and validate the model's response.

    Returns the stripped response if it matches ^(BENIGN|CWE-\\d{2,4})$,
    otherwise returns 'BENIGN' (label-collapsing step, Section 3.3).
    """
    cleaned = raw.strip().upper()
    if _OUTPUT_RE.match(cleaned):
        return cleaned
    return "BENIGN"


# ---------------------------------------------------------------------------
# API callers — one per provider
# ---------------------------------------------------------------------------
def call_openai(prompt: str, model_id: str) -> str:
    import openai
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        max_tokens=20,   # responses are always short: BENIGN or CWE-xxx
    )
    return response.choices[0].message.content or ""


def call_anthropic(prompt: str, model_id: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    message = client.messages.create(
        model=model_id,
        max_tokens=20,
        temperature=TEMPERATURE,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text if message.content else ""


def call_google(prompt: str, model_id: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel(
        model_id,
        generation_config=genai.types.GenerationConfig(
            temperature=TEMPERATURE,
            max_output_tokens=20,
        ),
    )
    response = model.generate_content(prompt)
    return response.text if response.text else ""


API_DISPATCH = {
    "openai":    call_openai,
    "anthropic": call_anthropic,
    "google":    call_google,
}


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
def evaluate(model_key: str, strategy: str, df: pd.DataFrame) -> list[dict]:
    """
    Run inference for a single (model, strategy) combination.

    Returns a list of result dicts:
      { "index": int, "ground_truth": str, "prediction": str, "raw_response": str }
    """
    cfg      = LLM_MODELS[model_key]
    api_fn   = API_DISPATCH[cfg["api"]]
    model_id = cfg["model_id"]

    # Sample few-shot examples once with the fixed seed — same set for all instances
    few_shot_examples = (
        sample_few_shot_examples(df, k=FEW_SHOT_K, seed=FEW_SHOT_SEED)
        if strategy == "few_shot"
        else []
    )

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
        else:
            raise ValueError(f"Unsupported strategy for LLMs: {strategy!r}. "
                             "Use 'one_shot' or 'few_shot'.")

        try:
            raw = api_fn(prompt, model_id)
        except Exception as exc:
            print(f"  [WARN] API error at index {idx}: {exc}. Retrying once …")
            time.sleep(5)
            try:
                raw = api_fn(prompt, model_id)
            except Exception:
                raw = ""

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
    parser = argparse.ArgumentParser(description="Evaluate proprietary LLMs on SVD-Benchmark.")
    parser.add_argument("--model",    choices=list(LLM_MODELS.keys()) + ["all"],
                        default="all", help="Model to evaluate")
    parser.add_argument("--strategy", choices=["one_shot", "few_shot", "all"],
                        default="all", help="Prompting strategy")
    args = parser.parse_args()

    df = load_dataset(DATASET_PATH)

    models     = list(LLM_MODELS.keys()) if args.model    == "all" else [args.model]
    strategies = ["one_shot", "few_shot"] if args.strategy == "all" else [args.strategy]

    for model_key in models:
        for strategy in strategies:
            print(f"\n=== {LLM_MODELS[model_key]['paper_name']} | {strategy} ===")
            results = evaluate(model_key, strategy, df)
            save_results(results, model_key, strategy)


if __name__ == "__main__":
    main()
