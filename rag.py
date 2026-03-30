"""
rag.py — SimCSE-based Retrieval-Augmented Generation (RAG).

Implements Section 3.4 of the paper:
  "A CWE-Aware Benchmark and Comparative Evaluation of LLMs for Java
   Software Vulnerability Detection"

Key design decisions from the paper:
  - Embeddings: SimCSE (princeton-nlp/sup-simcse-roberta-large), NOT
    CodeBERT or UniXcoder.  SimCSE's contrastive training objective
    directly optimises cosine-similarity, which aligns with the retrieval
    objective.  In a held-out 500-snippet comparison, SimCSE achieved
    68% top-1 CWE-match accuracy vs 61% for CodeBERT (7pp advantage).
  - Similarity metric: cosine similarity.
  - Retrieval source: the RAG pool (test='yes' rows) — never the instance
    being evaluated, to prevent retrieval leakage.
  - Retrieved examples replace the k=3 random examples in the Few-Shot
    template (Figure 2, Configuration B).
  - RAG is applied ONLY to SLMs (CodeGemma, Qwen2.5-Coder, Llama 3.2).
    Proprietary LLMs are confined to One-Shot and Few-Shot only (Section 3.4).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

from config import SIMCSE_MODEL, RAG_TOP_K


class SimCSERetriever:
    """
    Encodes snippets with SimCSE and retrieves the top-k most similar
    examples from the RAG pool using cosine similarity.
    """

    def __init__(self, model_name: str = SIMCSE_MODEL, top_k: int = RAG_TOP_K):
        self.top_k     = top_k
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModel.from_pretrained(model_name)
        self.model.eval()

        self._pool_embeddings: np.ndarray | None = None
        self._pool_records: list[dict]           = []

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------
    def _encode(self, texts: list[str]) -> np.ndarray:
        """
        Encode a list of texts into normalised SimCSE embeddings.

        Uses mean pooling over the last hidden state, then L2 normalises
        so that cosine similarity reduces to a dot product.
        """
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self.model(**encoded)
        # Mean pool over token dimension
        attention_mask = encoded["attention_mask"]
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        embeddings = (token_embeddings * input_mask_expanded).sum(1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
        embeddings = embeddings.numpy()
        # L2 normalise
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.maximum(norms, 1e-9)

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------
    def build_index(self, pool_df: pd.DataFrame) -> None:
        """
        Pre-compute SimCSE embeddings for every snippet in the RAG pool.

        pool_df must be the output of data_utils.get_rag_pool() — i.e.,
        the test='yes' rows that serve as the knowledge base.
        """
        snippets = pool_df["Code Snippet"].tolist()
        self._pool_records = [
            {"code": row["Code Snippet"], "label": row["label"]}
            for _, row in pool_df.iterrows()
        ]
        print(f"[RAG] Encoding {len(snippets)} pool snippets with SimCSE …")
        self._pool_embeddings = self._encode(snippets)
        print("[RAG] Index ready.")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve(self, query_snippet: str) -> list[dict]:
        """
        Retrieve the top-k most semantically similar exemplars from the
        pool for a given target code snippet.

        Returns a list of dicts with keys 'code' and 'label', ready to
        be passed to prompts.build_rag_prompt().
        """
        if self._pool_embeddings is None:
            raise RuntimeError("Call build_index() before retrieve().")

        query_emb = self._encode([query_snippet])          # (1, d)
        # Cosine similarity = dot product after L2 normalisation
        scores    = (self._pool_embeddings @ query_emb.T).squeeze()  # (N,)
        top_k_idx = np.argsort(scores)[::-1][: self.top_k]

        return [self._pool_records[i] for i in top_k_idx]
