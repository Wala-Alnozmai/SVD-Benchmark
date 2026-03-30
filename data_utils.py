"""
data_utils.py — Dataset loading and split helpers.

CSV columns:
  CWE ID, Project Name, Vulnerable File, Programming Language,
  Line Number, Code Snippet, Exact Vulnerable Line, Description,
  Status, test

  Status : 'vulnerable' | 'benign'
  CWE ID : 'CWE-xxx'   for vulnerable rows, 'BENIGN' for benign rows
  test   : 'yes' = RAG retrieval pool (1,036 vulnerable exemplars)
           'no'  = training / main evaluation set (4,017 rows)
"""

import pandas as pd
from config import DATASET_PATH


def load_dataset(path: str = DATASET_PATH) -> pd.DataFrame:
    """Load the full SVD-Benchmark CSV and return a cleaned DataFrame."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["Status"]  = df["Status"].str.strip().str.lower()
    df["test"]    = df["test"].str.strip().str.lower()
    df["CWE ID"]  = df["CWE ID"].str.strip()
    df["label"]   = df["CWE ID"]   # convenience alias used by prompt / eval code
    return df


def get_eval_set(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return the evaluation set — all 5,054 instances.
    (Both test='yes' and test='no' rows are evaluated for LLM/SAST experiments.)
    """
    return df.reset_index(drop=True)


def get_rag_pool(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return the RAG retrieval pool.

    Per Section 3.4 of the paper, the RAG framework retrieves from a
    predefined knowledge base.  The test='yes' rows (1,036 vulnerable
    exemplars) serve as this pool; they are never in the query position
    during RAG evaluation, which avoids retrieval-leakage.
    """
    pool = df[df["test"] == "yes"].reset_index(drop=True)
    assert len(pool) > 0, "RAG pool is empty — check the 'test' column."
    return pool


def get_few_shot_pool(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return the pool from which k=3 few-shot examples are sampled.
    Uses test='no' rows (training split) to avoid test-set contamination.
    """
    return df[df["test"] == "no"].reset_index(drop=True)


def sample_few_shot_examples(df: pd.DataFrame, k: int = 3, seed: int = 42) -> list[dict]:
    """
    Sample k few-shot examples from df using a fixed random seed.

    The seed is held constant across ALL models and ALL test instances so
    that differences in Few-Shot performance are attributable to model
    capability rather than example-selection variance (Section 3.3).

    Returns a list of dicts with keys 'code' and 'label'.
    """
    pool = get_few_shot_pool(df)
    sampled = pool.sample(n=k, random_state=seed)
    return [
        {"code": row["Code Snippet"], "label": row["label"]}
        for _, row in sampled.iterrows()
    ]
