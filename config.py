"""
config.py — Shared constants, model identifiers, and hyperparameters.

All values are taken directly from the paper:
  "A CWE-Aware Benchmark and Comparative Evaluation of LLMs for Java
   Software Vulnerability Detection"
"""

import os

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "SVD-Benchmark.csv")
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "..", "results")

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
# Fixed seed used to sample the k=3 few-shot examples (held constant across
# all models and all test instances — see Section 3.3 of the paper).
FEW_SHOT_SEED = 42

# Number of few-shot examples prepended to the prompt (Section 3.3).
FEW_SHOT_K = 3

# Number of bootstrap resamples for confidence interval computation (Section 3.5).
N_BOOTSTRAP = 5_000

# ---------------------------------------------------------------------------
# Inference settings
# ---------------------------------------------------------------------------
# Temperature=0 for all models ensures deterministic, reproducible outputs.
TEMPERATURE = 0

# Top-k retrieved neighbours for RAG (Section 3.4).
RAG_TOP_K = 3

# SimCSE model used for RAG embeddings (Section 3.4).
SIMCSE_MODEL = "princeton-nlp/sup-simcse-roberta-large"

# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------
# Any model response NOT matching this regex is collapsed to BENIGN (Section 3.3).
OUTPUT_REGEX = r"^(BENIGN|CWE-\d{2,4})$"

# ---------------------------------------------------------------------------
# Proprietary LLM model identifiers (accessed via official APIs)
# ---------------------------------------------------------------------------
LLM_MODELS = {
    "chatgpt": {
        "api":        "openai",
        "model_id":   "ChatGPT-5.2",   # ChatGPT-5.2 — closest available API string
        "paper_name": "ChatGPT-5.2",
    },
    "claude": {
        "api":        "anthropic",
        "model_id":   "claude-sonnet-4-5",   # Claude 4.5 Sonnet (from bib entry)
        "paper_name": "Claude 4.5 Sonnet",
    },
    "gemini": {
        "api":        "google",
        "model_id":   "gemini-3.0-flash",    # Gemini 3.0 Flash
        "paper_name": "Gemini 3.0 Flash",
    },
}

# ---------------------------------------------------------------------------
# Open-source SLM model identifiers (served locally via Ollama)
# Hardware: Apple MacBook Pro M3 Pro, 18 GB RAM (Section 3.2)
# ---------------------------------------------------------------------------
SLM_MODELS = {
    "codegemma": {
        "ollama_tag": "codegemma:2b",        # CodeGemma 2B (Google DeepMind)
        "paper_name": "CodeGemma",
        "params":     "2B",
    },
    "qwen": {
        "ollama_tag": "qwen2.5-coder:1.5b",  # Qwen2.5-Coder 1.5B (Alibaba)
        "paper_name": "Qwen2.5-Coder",
        "params":     "1.5B",
    },
    "llama": {
        "ollama_tag": "llama3.2:1b",         # Llama 3.2 1B (Meta)
        "paper_name": "Llama 3.2",
        "params":     "1B",
    },
}

# ---------------------------------------------------------------------------
# Prompting strategies
# ---------------------------------------------------------------------------
STRATEGIES = {
    "llm":  ["one_shot", "few_shot"],   # Proprietary LLMs: One-Shot + Few-Shot ICL only
    "slm":  ["one_shot", "few_shot", "rag"],  # SLMs: all three strategies
}
